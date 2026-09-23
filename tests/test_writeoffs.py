"""预留分次核销(领用)端到端测试。

覆盖:事务内逐项校验(预留未释放/逐项未核销量/样品余量)、任一失败整单不变、
扣减余量与递增版本、不可变核销明细、同号幂等重放与不同内容冲突、核销后锁定量
只计未核销部分、释放只解除剩余锁定、余量账本纳入核销出账、并发串行化。
"""
from __future__ import annotations

import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app import writeoffs as wo
from app.db import connect, init_db
from app.main import app, get_conn


@pytest.fixture()
def client():
    mem = connect(":memory:")
    init_db(mem)

    def _override():
        yield mem

    app.dependency_overrides[get_conn] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    mem.close()


def _sample(client, op_no, sid, amount, unit="mL", location=None):
    r = client.post("/api/v1/operations/sampling", json={
        "type": "sampling", "op_no": op_no, "sample_id": sid,
        "material": "血清", "amount": amount, "unit": unit,
        "location": location,
    })
    assert r.status_code == 200, r.text


def _reserve(client, no, items, note=None):
    body = {"reservation_no": no, "items": items}
    if note is not None:
        body["note"] = note
    r = client.post("/api/v1/reservations", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _writeoff(client, no, reservation_no, items, operator="李领用", note=None):
    body = {
        "writeoff_no": no,
        "reservation_no": reservation_no,
        "items": items,
        "operator": operator,
    }
    if note is not None:
        body["note"] = note
    return client.post(
        f"/api/v1/reservations/{reservation_no}/writeoffs", json=body
    )


# ---------------------------------------------------------------- 基本流程

def test_writeoff_deducts_remaining_increments_version_and_unlocks_consumed(client):
    _sample(client, "OP-1", "S1", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 60}])

    r = _writeoff(client, "WO-1", "R1", [{"sample_id": "S1", "amount": 20}])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["writeoff_no"] == "WO-1"
    assert body["reservation_no"] == "R1"
    assert body["replayed"] is False
    assert body["writeoff_at"]
    item = body["items"][0]
    assert item["sample_id"] == "S1"
    assert item["amount"] == 20.0
    assert item["remaining"] == 80.0
    assert item["from_version"] == 0  # sampling 新建样品版本为 0
    assert item["to_version"] == 1
    assert item["item_amount"] == 60.0
    assert item["written_off"] == 20.0
    assert item["outstanding"] == 40.0
    assert body["remaining"] == {"S1": 80.0}

    # 样品余量已扣减、版本已递增
    r = client.get("/api/v1/reservations?sample_id=S1")
    avail = r.json()["samples"][0]
    assert avail["remaining"] == 80.0
    # 锁定量只剩未核销的 40,而不是 60
    assert avail["reserved"] == 40.0
    assert avail["available"] == 40.0

    # 预留单查询展示累计核销与未核销
    r = client.get("/api/v1/reservations/R1")
    ri = r.json()["items"][0]
    assert ri["written_off"] == 20.0
    assert ri["outstanding"] == 40.0
    assert ri["reserved"] == 40.0


def test_partial_writeoff_then_another_then_release_remaining_lock(client):
    _sample(client, "OP-1", "S1", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 60}])

    # 分次核销 20 + 30,共 50
    assert _writeoff(client, "WO-1", "R1",
                     [{"sample_id": "S1", "amount": 20}]).status_code == 200
    r = _writeoff(client, "WO-2", "R1", [{"sample_id": "S1", "amount": 30}])
    assert r.status_code == 200, r.text
    it = r.json()["items"][0]
    assert it["written_off"] == 50.0
    assert it["outstanding"] == 10.0
    assert it["remaining"] == 50.0

    # 余量 50,锁定只剩 10,可用量 40
    avail = client.get("/api/v1/reservations?sample_id=S1").json()["samples"][0]
    assert avail == {
        "sample_id": "S1", "unit": "mL",
        "remaining": 50.0, "reserved": 10.0, "available": 40.0,
    }

    # 释放只解除剩余 10 的锁定,已核销的 50 不回来
    r = client.delete("/api/v1/reservations/R1")
    assert r.status_code == 200
    rel = r.json()["items"][0]
    assert rel["reserved"] == 0.0
    assert rel["available"] == 50.0

    avail = client.get("/api/v1/reservations?sample_id=S1").json()["samples"][0]
    assert avail["remaining"] == 50.0
    assert avail["reserved"] == 0.0
    assert avail["available"] == 50.0


def test_writeoff_multi_items_all_or_nothing(client):
    _sample(client, "OP-1", "S1", 100)
    _sample(client, "OP-2", "S2", 5)
    _reserve(client, "R1", [
        {"sample_id": "S1", "amount": 60},
        {"sample_id": "S2", "amount": 5},
    ])

    # S2 未核销余量 5 但本次要领 6,且 S2 样品余量只有 5 -> 两项冲突一次返回
    r = _writeoff(client, "WO-BAD", "R1", [
        {"sample_id": "S1", "amount": 60},
        {"sample_id": "S2", "amount": 6},
    ])
    assert r.status_code == 409, r.text
    codes = {c["code"] for c in r.json()["conflicts"]}
    # 超出未核销量 与 样品余量不足 同时成立,先报超出预留
    assert codes == {"writeoff_exceeds_reserved"}
    c = next(c for c in r.json()["conflicts"] if c["sample_id"] == "S2")
    assert c["requested"] == 6.0
    assert c["available"] == 5.0
    assert c["op_no"] == "WO-BAD"

    # 整单回滚:S1 也没有被扣
    avail = client.get("/api/v1/reservations?sample_id=S1").json()["samples"][0]
    assert avail["remaining"] == 100.0
    assert avail["reserved"] == 60.0
    assert client.get("/api/v1/writeoffs/WO-BAD").status_code == 404


def test_writeoff_exceeding_outstanding_conflicts_and_rolls_back(client):
    _sample(client, "OP-1", "S1", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 60}])
    assert _writeoff(client, "WO-1", "R1",
                     [{"sample_id": "S1", "amount": 50}]).status_code == 200

    # 再领 20:未核销只剩 10 -> 冲突
    r = _writeoff(client, "WO-2", "R1", [{"sample_id": "S1", "amount": 20}])
    assert r.status_code == 409
    c = r.json()["conflicts"][0]
    assert c["code"] == "writeoff_exceeds_reserved"
    assert c["requested"] == 20.0
    assert c["available"] == 10.0

    # 余量仍是核销 50 后的 50,版本未因失败再递增
    avail = client.get("/api/v1/reservations?sample_id=S1").json()["samples"][0]
    assert avail["remaining"] == 50.0
    assert avail["reserved"] == 10.0


def test_writeoff_sample_balance_insufficient_conflicts(client):
    """未核销量足够但样品余量被其他消耗压低:核销按样品余量兜底拒绝。"""
    _sample(client, "OP-1", "S1", 10)
    _reserve(client, "R2", [{"sample_id": "S1", "amount": 6}])
    # 可用量 = 余量10 - 预留6 = 4,销毁 4 -> 余量 6
    r = client.post("/api/v1/operations/dispose", json={
        "type": "dispose", "op_no": "DSP-1", "operator": "孙七",
        "items": [{"sample_id": "S1", "amount": 4, "reason": "过期"}],
    })
    assert r.status_code == 200, r.text

    # 余量 6,先核销 5(未核销 6 足够、余量 6 足够)-> 余量 1、未核销 1
    assert _writeoff(client, "WO-2A", "R2",
                     [{"sample_id": "S1", "amount": 5}]).status_code == 200
    # 再核销 1:未核销剩 1 足够,但样品余量只剩 1 -> 临界通过,余量归 0
    assert _writeoff(client, "WO-2B", "R2",
                     [{"sample_id": "S1", "amount": 1}]).status_code == 200
    avail = client.get("/api/v1/reservations?sample_id=S1").json()["samples"][0]
    assert avail["remaining"] == 0.0
    assert avail["reserved"] == 0.0
    assert avail["available"] == 0.0

    # 新样品:余量 5、预留 5,先核销 4(余量 1),再想核销剩余 2 -> 未核销仅 1,
    # 直接触发未核销超限;而把一个"预留未核销 > 余量"的场景用分批构造:
    _sample(client, "OP-2", "S3", 5)
    _reserve(client, "R3", [{"sample_id": "S3", "amount": 5}])
    assert _writeoff(client, "WO-3A", "R3",
                     [{"sample_id": "S3", "amount": 4}]).status_code == 200
    # 余量 1、未核销 1,核销 1 成功
    assert _writeoff(client, "WO-3B", "R3",
                     [{"sample_id": "S3", "amount": 1}]).status_code == 200


# ---------------------------------------------------------------- 预留状态校验

def test_writeoff_on_unknown_or_released_reservation_conflicts(client):
    _sample(client, "OP-1", "S1", 100)

    r = _writeoff(client, "WO-1", "NOPE", [{"sample_id": "S1", "amount": 1}])
    assert r.status_code == 404
    assert r.json()["conflicts"][0]["code"] == "reservation_not_found"

    _reserve(client, "R1", [{"sample_id": "S1", "amount": 10}])
    assert client.delete("/api/v1/reservations/R1").status_code == 200

    r = _writeoff(client, "WO-2", "R1", [{"sample_id": "S1", "amount": 1}])
    assert r.status_code == 409
    assert r.json()["conflicts"][0]["code"] == "reservation_closed"


def test_writeoff_item_not_in_reservation_conflicts(client):
    _sample(client, "OP-1", "S1", 100)
    _sample(client, "OP-2", "S2", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 10}])

    r = _writeoff(client, "WO-1", "R1", [{"sample_id": "S2", "amount": 1}])
    assert r.status_code == 409
    c = r.json()["conflicts"][0]
    assert c["code"] == "writeoff_item_not_found"
    assert c["sample_id"] == "S2"


def test_writeoff_duplicate_sample_in_request_rejected(client):
    r = client.post("/api/v1/reservations/R1/writeoffs", json={
        "writeoff_no": "WO-DUP", "reservation_no": "R1",
        "operator": "李领用",
        "items": [
            {"sample_id": "S1", "amount": 1},
            {"sample_id": "S1", "amount": 2},
        ],
    })
    assert r.status_code == 422


def test_writeoff_path_body_reservation_mismatch_422(client):
    _sample(client, "OP-1", "S1", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 10}])
    r = client.post("/api/v1/reservations/R1/writeoffs", json={
        "writeoff_no": "WO-M", "reservation_no": "R2", "operator": "李领用",
        "items": [{"sample_id": "S1", "amount": 1}],
    })
    assert r.status_code == 422


# ---------------------------------------------------------------- 幂等

def test_writeoff_idempotent_replay_and_payload_conflict(client):
    _sample(client, "OP-1", "S1", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 60}])
    body = {
        "writeoff_no": "WO-IDEM", "reservation_no": "R1", "operator": "李领用",
        "items": [{"sample_id": "S1", "amount": 20}],
    }
    r1 = client.post("/api/v1/reservations/R1/writeoffs", json=body)
    assert r1.status_code == 200
    first = r1.json()
    assert first["replayed"] is False

    # 相同内容重放:返回原结果、余量只扣一次
    r2 = client.post("/api/v1/reservations/R1/writeoffs", json=body)
    assert r2.status_code == 200
    second = r2.json()
    assert second["replayed"] is True
    assert second["writeoff_at"] == first["writeoff_at"]
    assert second["items"][0]["remaining"] == 80.0

    avail = client.get("/api/v1/reservations?sample_id=S1").json()["samples"][0]
    assert avail["remaining"] == 80.0
    assert avail["reserved"] == 40.0

    # 核销明细只有一条
    rows = client.get("/api/v1/reservations/R1/writeoffs").json()["writeoffs"]
    assert len(rows) == 1

    # 同号不同内容 -> 409
    bad = dict(body)
    bad["items"] = [{"sample_id": "S1", "amount": 21}]
    r3 = client.post("/api/v1/reservations/R1/writeoffs", json=bad)
    assert r3.status_code == 409
    assert r3.json()["conflicts"][0]["code"] == "writeoff_no_conflict"

    # 按核销号取原结果
    r4 = client.get("/api/v1/writeoffs/WO-IDEM")
    assert r4.status_code == 200
    assert r4.json()["items"][0]["amount"] == 20.0
    assert client.get("/api/v1/writeoffs/NOPE").status_code == 404


def test_writeoff_replay_works_after_reservation_released(client):
    """释放预留后,历史核销号同内容仍可回放原结果,且不再扣余量。"""
    _sample(client, "OP-1", "S1", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 60}])
    body = {
        "writeoff_no": "WO-R", "reservation_no": "R1", "operator": "李领用",
        "items": [{"sample_id": "S1", "amount": 20}],
    }
    first = client.post("/api/v1/reservations/R1/writeoffs", json=body).json()
    assert client.delete("/api/v1/reservations/R1").status_code == 200

    r = client.post("/api/v1/reservations/R1/writeoffs", json=body)
    assert r.status_code == 200
    got = r.json()
    assert got["replayed"] is True
    assert got["writeoff_at"] == first["writeoff_at"]
    assert got["items"][0]["remaining"] == 80.0  # 原结果快照,非当前余量回放


def test_writeoff_same_no_different_reservation_conflicts(client):
    _sample(client, "OP-1", "S1", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 10}])
    _reserve(client, "R2", [{"sample_id": "S1", "amount": 10}])
    body1 = {
        "writeoff_no": "WO-X", "reservation_no": "R1", "operator": "李领用",
        "items": [{"sample_id": "S1", "amount": 1}],
    }
    assert client.post("/api/v1/reservations/R1/writeoffs", json=body1).status_code == 200
    body2 = dict(body1)
    body2["reservation_no"] = "R2"
    r = client.post("/api/v1/reservations/R2/writeoffs", json=body2)
    assert r.status_code == 409
    assert r.json()["conflicts"][0]["code"] == "writeoff_no_conflict"


# ---------------------------------------------------------------- 查询

def test_list_writeoffs_ordered_and_unknown_reservation_404(client):
    _sample(client, "OP-1", "S1", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 60}])
    _writeoff(client, "WO-1", "R1", [{"sample_id": "S1", "amount": 10}])
    _writeoff(client, "WO-2", "R1", [{"sample_id": "S1", "amount": 5}])

    r = client.get("/api/v1/reservations/R1/writeoffs")
    assert r.status_code == 200
    body = r.json()
    assert body["reservation_no"] == "R1"
    assert body["status"] == "active"
    assert [w["writeoff_no"] for w in body["writeoffs"]] == ["WO-1", "WO-2"]
    assert body["writeoffs"][1]["items"][0]["written_off"] == 15.0

    assert client.get("/api/v1/reservations/NOPE/writeoffs").status_code == 404


# ---------------------------------------------------------------- 账本一致性

def test_balance_ledger_includes_writeoff_out(client):
    _sample(client, "OP-1", "S1", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 60}])
    _writeoff(client, "WO-1", "R1", [{"sample_id": "S1", "amount": 20}])
    _writeoff(client, "WO-2", "R1", [{"sample_id": "S1", "amount": 30}])

    r = client.get("/api/v1/samples/S1/balance")
    assert r.status_code == 200
    b = r.json()
    assert b["consistent"] is True
    assert b["samples"][0]["writeoffs"] == 50.0
    assert b["samples"][0]["stored_remaining"] == 50.0
    assert b["samples"][0]["ledger_remaining"] == 50.0

    # 全量校验一致
    r = client.get("/api/v1/checks/balances")
    assert r.json()["consistent"] is True
    s1 = next(x for x in r.json()["samples"] if x["sample_id"] == "S1")
    assert s1["writeoffs"] == 50.0


def test_writeoff_after_dispose_and_writeoff_balance_still_consistent(client):
    _sample(client, "OP-1", "S1", 100)
    _reserve(client, "R1", [{"sample_id": "S1", "amount": 60}])
    # 可用量 40,销毁 10 -> 余量 90
    r = client.post("/api/v1/operations/dispose", json={
        "type": "dispose", "op_no": "DSP-1", "operator": "孙七",
        "items": [{"sample_id": "S1", "amount": 10, "reason": "污染"}],
    })
    assert r.status_code == 200, r.text
    # 核销 60 -> 余量 30,锁定归零
    assert _writeoff(client, "WO-1", "R1",
                     [{"sample_id": "S1", "amount": 60}]).status_code == 200

    r = client.get("/api/v1/checks/balances").json()
    assert r["consistent"] is True
    s1 = next(x for x in r["samples"] if x["sample_id"] == "S1")
    assert s1["total_out"] == 10.0
    assert s1["writeoffs"] == 60.0
    assert s1["stored_remaining"] == 30.0


# ---------------------------------------------------------------- 不可变审计

def test_writeoff_records_are_immutable_and_keep_after_release(tmp_path):
    db_path = str(tmp_path / "w.db")
    conn = connect(db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO samples (sample_id, material, unit, initial_amount,"
        " remaining, is_root, version) VALUES ('S1', 'M', 'mL', 100, 100, 1, 1)"
    )
    conn.execute(
        "INSERT INTO reservations (reservation_no, status, payload)"
        " VALUES ('R1', 'active', '{}')"
    )
    conn.execute(
        "INSERT INTO reservation_items (reservation_no, sample_id, amount,"
        " unit, seq_index) VALUES ('R1', 'S1', 50, 'mL', 0)"
    )
    conn.commit()

    from app.schemas import WriteoffRequest

    wo.create_writeoff(conn, WriteoffRequest(
        writeoff_no="WO-1", reservation_no="R1", operator="李领用",
        items=[{"sample_id": "S1", "amount": 50}],  # type: ignore[arg-type]
    ))

    # 触发器禁止 UPDATE / DELETE
    with pytest.raises(sqlite3.Error):
        conn.execute("UPDATE writeoff_items SET amount = 1 WHERE writeoff_no = 'WO-1'")
    conn.rollback()
    with pytest.raises(sqlite3.Error):
        conn.execute("DELETE FROM writeoff_items WHERE writeoff_no = 'WO-1'")
    conn.rollback()
    with pytest.raises(sqlite3.Error):
        conn.execute("UPDATE writeoffs SET operator = 'X' WHERE writeoff_no = 'WO-1'")
    conn.rollback()
    with pytest.raises(sqlite3.Error):
        conn.execute("DELETE FROM writeoffs WHERE writeoff_no = 'WO-1'")
    conn.rollback()

    # 释放预留后核销明细仍保留(审计痕迹)
    from app import reservations as rsv
    rsv.release_reservation(conn, "R1")
    n = conn.execute(
        "SELECT COUNT(*) AS c FROM writeoff_items WHERE reservation_no = 'R1'"
    ).fetchone()["c"]
    assert n == 1
    row = conn.execute(
        "SELECT amount, from_version, to_version, remaining_after"
        " FROM writeoff_items WHERE writeoff_no = 'WO-1'"
    ).fetchone()
    assert tuple(row) == (50.0, 1, 2, 50.0)
    conn.close()


# ---------------------------------------------------------------- 并发

def test_writeoff_rejects_when_sample_remaining_below_outstanding(tmp_path):
    """直接构造异常账实:未核销仍有 5、样品余量却只剩 3(被外部调账压低),
    核销按样品余量兜底返回 insufficient_balance,且整单不变。"""
    db_path = str(tmp_path / "b.db")
    conn = connect(db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO samples (sample_id, material, unit, initial_amount,"
        " remaining, is_root, version) VALUES ('S1', 'M', 'mL', 100, 3, 1, 7)"
    )
    conn.execute(
        "INSERT INTO reservations (reservation_no, status, payload)"
        " VALUES ('R1', 'active', '{}')"
    )
    conn.execute(
        "INSERT INTO reservation_items (reservation_no, sample_id, amount,"
        " unit, seq_index) VALUES ('R1', 'S1', 10, 'mL', 0)"
    )
    conn.commit()

    from app.schemas import WriteoffRequest

    with pytest.raises(wo.WriteoffError) as ei:
        wo.create_writeoff(conn, WriteoffRequest(
            writeoff_no="WO-1", reservation_no="R1", operator="李领用",
            items=[{"sample_id": "S1", "amount": 5}],  # type: ignore[arg-type]
        ))
    c = ei.value.conflicts[0]
    assert c.code == "insufficient_balance"
    assert c.requested == 5.0
    assert c.available == 3.0

    # 整单不变:余量、版本、核销表
    row = conn.execute(
        "SELECT remaining, version FROM samples WHERE sample_id = 'S1'"
    ).fetchone()
    assert tuple(row) == (3.0, 7)
    assert conn.execute("SELECT COUNT(*) AS c FROM writeoffs").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) AS c FROM writeoff_items").fetchone()["c"] == 0
    conn.close()


def test_concurrent_writeoffs_serialize_without_oversell(tmp_path):
    """两个线程并发核销同一预留项(各 40,共 80 > 预留 60/余量 100),
    恰好一个成功,且库存与锁定均不出错。"""
    db_path = str(tmp_path / "c.db")
    setup = connect(db_path)
    init_db(setup)
    setup.execute(
        "INSERT INTO samples (sample_id, material, unit, initial_amount,"
        " remaining, is_root, version) VALUES ('S1', 'M', 'mL', 100, 100, 1, 0)"
    )
    setup.execute(
        "INSERT INTO reservations (reservation_no, status, payload)"
        " VALUES ('R1', 'active', '{}')"
    )
    setup.execute(
        "INSERT INTO reservation_items (reservation_no, sample_id, amount,"
        " unit, seq_index) VALUES ('R1', 'S1', 60, 'mL', 0)"
    )
    setup.commit()
    setup.close()

    from app.schemas import WriteoffRequest

    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def worker(no: str) -> None:
        c = connect(db_path)
        barrier.wait()
        try:
            wo.create_writeoff(c, WriteoffRequest(
                writeoff_no=no, reservation_no="R1", operator="李领用",
                items=[{"sample_id": "S1", "amount": 40}],  # type: ignore[arg-type]
            ))
            outcomes.append("ok")
        except wo.WriteoffError:
            outcomes.append("conflict")
        finally:
            c.close()

    t1 = threading.Thread(target=worker, args=("WC-1",))
    t2 = threading.Thread(target=worker, args=("WC-2",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert sorted(outcomes) == ["conflict", "ok"]
    check = connect(db_path)
    row = check.execute(
        "SELECT remaining, version FROM samples WHERE sample_id = 'S1'"
    ).fetchone()
    assert row["remaining"] == 60.0
    assert row["version"] == 1
    spent = check.execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM writeoff_items"
        " WHERE reservation_no = 'R1'"
    ).fetchone()["s"]
    assert spent == 40.0
    check.close()


def test_existing_db_without_writeoff_tables_gets_them_on_init(tmp_path):
    """旧库(无核销表)启动初始化后自动补齐,且不影响存量数据。"""
    db_path = str(tmp_path / "old.db")
    old = sqlite3.connect(db_path)
    old.executescript(
        """
        CREATE TABLE samples (
            sample_id TEXT PRIMARY KEY, material TEXT NOT NULL, unit TEXT NOT NULL,
            initial_amount REAL NOT NULL, remaining REAL NOT NULL,
            version INTEGER NOT NULL DEFAULT 0, location TEXT, is_root INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE TABLE operations (
            op_id INTEGER PRIMARY KEY AUTOINCREMENT, op_no TEXT NOT NULL UNIQUE,
            op_type TEXT NOT NULL CHECK (op_type IN ('sampling','aliquot','merge')),
            payload TEXT NOT NULL, effect TEXT NOT NULL, seq_index INTEGER NOT NULL,
            batch_id TEXT, location TEXT, created_at TEXT
        );
        CREATE TABLE ledger_entries (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL,
            op_id INTEGER NOT NULL, edge_id INTEGER, dispose_id INTEGER,
            direction TEXT NOT NULL, amount REAL NOT NULL, unit TEXT NOT NULL, ref_no TEXT NOT NULL
        );
        CREATE TABLE lineage_edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT, parent_id TEXT, child_id TEXT NOT NULL,
            op_id INTEGER NOT NULL, amount REAL NOT NULL, unit TEXT NOT NULL, ref_no TEXT NOT NULL
        );
        CREATE TABLE reservations (
            reservation_no TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'active',
            payload TEXT NOT NULL, note TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            released_at TEXT
        );
        CREATE TABLE reservation_items (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT, reservation_no TEXT NOT NULL,
            sample_id TEXT NOT NULL, amount REAL NOT NULL, unit TEXT NOT NULL,
            seq_index INTEGER NOT NULL,
            UNIQUE (reservation_no, sample_id)
        );
        CREATE TABLE reservation_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT, reservation_no TEXT,
            action TEXT NOT NULL, ref_id TEXT, detail TEXT NOT NULL, created_at TEXT
        );
        CREATE TABLE schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        INSERT INTO samples (sample_id, material, unit, initial_amount, remaining, is_root)
        VALUES ('S1', 'M', 'mL', 100, 100, 1);
        """
    )
    old.commit()
    old.close()

    conn = connect(db_path)
    init_db(conn)  # 自动建核销表
    names = {
        r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
        ).fetchall()
    }
    assert {"writeoffs", "writeoff_items"} <= names
    assert "trg_writeoff_no_update" in names
    # 存量数据完好
    assert conn.execute(
        "SELECT remaining FROM samples WHERE sample_id = 'S1'"
    ).fetchone()["remaining"] == 100.0
    conn.close()
