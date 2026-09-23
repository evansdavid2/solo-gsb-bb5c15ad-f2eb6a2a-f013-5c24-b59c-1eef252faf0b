"""预留分次核销(领用)服务:对已有未释放预留单分次领用样品。

规则
----
- 每次提交核销号 `writeoff_no`、预留号 `reservation_no` 与若干样品领用量,
  服务在单个 `BEGIN IMMEDIATE` 事务内逐项校验:
    1. 预留单存在且**未释放**(已释放单只能回放其历史核销结果,不能新增);
    2. 样品在该预留单中,且本次领用量不超过其**未核销量**
       (预留量 − 该单累计已核销,即仍被本单锁定的量);
    3. 样品当前余量足够本次领用。
  任一失败**整单不变**(不扣余量、不递增版本、不写核销明细),响应
  `conflicts[]` 一次返回全部冲突明细。
- 全部通过后:逐项扣减样品余量并**递增样品版本**,写不可变核销明细
  `writeoff_items`(触发器禁止 UPDATE/DELETE);被核销部分此后不再计入
  该预留单的锁定量,因此释放预留时只解除剩余(未核销)锁定。
- 核销出库(非谱系出账)纳入余量账本重算,`/api/v1/checks/balances`
  中各样品的 `writeoffs` 字段给出累计核销合计。

幂等
----
同一核销号 + 同一请求体重放返回首次结果(`replayed: true`),
不重复出库、不重复写明细;同号不同内容返回 `409 writeoff_no_conflict`。
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from . import schemas as s

_EPS = 1e-9  # 运算时浮点比较


class WriteoffError(Exception):
    """核销业务冲突,携带结构化冲突列表。"""

    def __init__(self, conflicts: list[s.Conflict]):
        self.conflicts = conflicts
        super().__init__(conflicts[0].message if conflicts else "writeoff conflict")


def _r(x: float) -> float:
    """统一保留 9 位小数,消除浮点尾巴。"""
    return round(float(x), 9)


def _conflict(
    code: str,
    message: str,
    *,
    writeoff_no: Optional[str] = None,
    reservation_no: Optional[str] = None,
    sample_id: Optional[str] = None,
    requested: Optional[float] = None,
    available: Optional[float] = None,
) -> s.Conflict:
    return s.Conflict(
        code=code,
        message=message,
        op_no=writeoff_no,
        sample_id=sample_id,
        requested=requested,
        available=available,
    )


def _get_header(
    conn: sqlite3.Connection, writeoff_no: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM writeoffs WHERE writeoff_no = ?", (writeoff_no,)
    ).fetchone()


def _load_result(row: sqlite3.Row, *, replayed: bool) -> s.WriteoffResult:
    """从核销头表保存的结果快照还原幂等回放结果。"""
    data = json.loads(row["result"])
    return s.WriteoffResult(
        writeoff_no=row["writeoff_no"],
        reservation_no=row["reservation_no"],
        operator=data["operator"],
        note=data.get("note"),
        items=[s.WriteoffItemOut(**i) for i in data["items"]],
        remaining=data["remaining"],
        writeoff_at=row["created_at"],
        created_at=row["created_at"],
        replayed=replayed,
    )


# ---------------------------------------------------------------- 提交核销

def create_writeoff(
    conn: sqlite3.Connection, req: s.WriteoffRequest
) -> s.WriteoffResult:
    """单事务内校验预留未释放、逐项未核销量与样品余量,全部通过才扣减出库。"""
    payload = req.model_dump_json()

    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = _get_header(conn, req.writeoff_no)
        if existing is not None:
            if existing["payload"] != payload:
                raise WriteoffError([
                    _conflict(
                        "writeoff_no_conflict",
                        f"核销号 {req.writeoff_no} 已存在但请求内容不同,"
                        "同一核销号必须对应同一次核销(幂等键冲突)",
                        writeoff_no=req.writeoff_no,
                        reservation_no=req.reservation_no,
                    )
                ])
            result = _load_result(existing, replayed=True)
            conn.execute("COMMIT")
            return result

        # 1) 预留单必须存在且未释放
        reservation = conn.execute(
            "SELECT * FROM reservations WHERE reservation_no = ?",
            (req.reservation_no,),
        ).fetchone()
        if reservation is None:
            raise WriteoffError([
                _conflict(
                    "reservation_not_found",
                    f"预留号 {req.reservation_no} 不存在,无法核销",
                    writeoff_no=req.writeoff_no,
                    reservation_no=req.reservation_no,
                )
            ])
        if reservation["status"] == "released":
            raise WriteoffError([
                _conflict(
                    "reservation_closed",
                    f"预留号 {req.reservation_no} 已释放,不能再核销",
                    writeoff_no=req.writeoff_no,
                    reservation_no=req.reservation_no,
                )
            ])

        # 2) 取预留明细(预留量/单位)、累计已核销、样品当前余量
        items = {
            r["sample_id"]: r
            for r in conn.execute(
                "SELECT * FROM reservation_items WHERE reservation_no = ?",
                (req.reservation_no,),
            ).fetchall()
        }
        spent_rows = conn.execute(
            "SELECT sample_id, COALESCE(SUM(amount), 0) AS spent"
            " FROM writeoff_items WHERE reservation_no = ? GROUP BY sample_id",
            (req.reservation_no,),
        ).fetchall()
        spent = {r["sample_id"]: _r(r["spent"]) for r in spent_rows}

        ids = list(dict.fromkeys(i.sample_id for i in req.items))  # model 已保证不重复
        qmarks = ",".join("?" * len(ids))
        samples = {
            r["sample_id"]: r
            for r in conn.execute(
                f"SELECT * FROM samples WHERE sample_id IN ({qmarks})", ids
            ).fetchall()
        }

        # 3) 先做全部静态校验,收集逐项冲突后一次性抛出
        conflicts: list[s.Conflict] = []
        plan: list[tuple[str, float]] = []  # (sample_id, amount)
        for item in req.items:
            amount = _r(item.amount)
            rsv_item = items.get(item.sample_id)
            if rsv_item is None:
                conflicts.append(_conflict(
                    "writeoff_item_not_found",
                    f"样品 {item.sample_id} 不在预留单 {req.reservation_no} 中,"
                    "不能核销该预留下的领用",
                    writeoff_no=req.writeoff_no,
                    reservation_no=req.reservation_no,
                    sample_id=item.sample_id,
                    requested=amount,
                ))
                continue
            already = spent.get(item.sample_id, 0.0)
            outstanding = _r(rsv_item["amount"] - already)
            if amount > outstanding + _EPS:
                conflicts.append(_conflict(
                    "writeoff_exceeds_reserved",
                    f"样品 {item.sample_id} 在预留单 {req.reservation_no} 中预留"
                    f" {_r(rsv_item['amount'])} {rsv_item['unit']},已核销 {already}"
                    f" {rsv_item['unit']},未核销仅 {outstanding} {rsv_item['unit']},"
                    f"本次领用 {amount} {rsv_item['unit']} 超出未核销(仍锁定)量",
                    writeoff_no=req.writeoff_no,
                    reservation_no=req.reservation_no,
                    sample_id=item.sample_id,
                    requested=amount,
                    available=outstanding,
                ))
                continue
            sample = samples.get(item.sample_id)
            if sample is None:
                # 理论上被外键兜底:预留明细引用的样品应存在;防御性报错
                conflicts.append(_conflict(
                    "sample_not_found",
                    f"样品 {item.sample_id} 不存在,无法核销",
                    writeoff_no=req.writeoff_no,
                    reservation_no=req.reservation_no,
                    sample_id=item.sample_id,
                    requested=amount,
                ))
                continue
            remaining = _r(sample["remaining"])
            if amount > remaining + _EPS:
                conflicts.append(_conflict(
                    "insufficient_balance",
                    f"样品 {item.sample_id} 当前余量 {remaining} {sample['unit']}"
                    f" 不足,本次核销领用需要 {amount} {sample['unit']}",
                    writeoff_no=req.writeoff_no,
                    reservation_no=req.reservation_no,
                    sample_id=item.sample_id,
                    requested=amount,
                    available=remaining,
                ))
                continue
            plan.append((item.sample_id, amount))
        if conflicts:
            raise WriteoffError(conflicts)

        # 4) 组装逐项结果(扣减后余量、累计核销、剩余锁定、前后版本)
        out_items: list[s.WriteoffItemOut] = []
        remaining_view: dict[str, float] = {}
        for idx, (sample_id, amount) in enumerate(plan):
            sample = samples[sample_id]
            rsv_item = items[sample_id]
            already = spent.get(sample_id, 0.0)
            remaining_after = _r(sample["remaining"] - amount)
            written_after = _r(already + amount)
            outstanding_after = _r(rsv_item["amount"] - written_after)
            out_items.append(s.WriteoffItemOut(
                sample_id=sample_id,
                amount=amount,
                unit=sample["unit"],
                remaining=remaining_after,
                from_version=sample["version"],
                to_version=sample["version"] + 1,
                item_amount=_r(rsv_item["amount"]),
                written_off=written_after,
                outstanding=outstanding_after,
            ))
            remaining_view[sample_id] = remaining_after

        result = s.WriteoffResult(
            writeoff_no=req.writeoff_no,
            reservation_no=req.reservation_no,
            operator=req.operator,
            note=req.note,
            items=out_items,
            remaining=remaining_view,
            writeoff_at="",  # 落库后用头表 created_at 覆盖
        )
        result_snapshot = result.model_dump()

        # 5) 落库:先取与触发器默认值一致的时间戳,写头表拿 writeoff_id,
        #    再逐项扣余量/递增版本、写不可变核销明细
        created_at = conn.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now') AS ts"
        ).fetchone()["ts"]
        cur = conn.execute(
            "INSERT INTO writeoffs (writeoff_no, reservation_no, operator,"
            " payload, result, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (req.writeoff_no, req.reservation_no, req.operator, payload,
             json.dumps(result_snapshot, ensure_ascii=False), created_at),
        )
        writeoff_id = cur.lastrowid
        for idx, (sample_id, amount) in enumerate(plan):
            out = out_items[idx]
            conn.execute(
                "UPDATE samples SET remaining = remaining - ?,"
                " version = version + 1 WHERE sample_id = ?",
                (amount, sample_id),
            )
            conn.execute(
                "INSERT INTO writeoff_items (writeoff_id, writeoff_no,"
                " reservation_no, sample_id, seq_index, amount, unit,"
                " from_version, to_version, remaining_after, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (writeoff_id, req.writeoff_no, req.reservation_no, sample_id,
                 idx, amount, out.unit, out.from_version, out.to_version,
                 out.remaining, created_at),
            )

        result.writeoff_at = created_at
        result.created_at = created_at
        conn.execute("COMMIT")
        return result
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------- 查询

def get_writeoff(
    conn: sqlite3.Connection, writeoff_no: str
) -> s.WriteoffResult:
    row = _get_header(conn, writeoff_no)
    if row is None:
        raise WriteoffError([
            _conflict(
                "writeoff_not_found",
                f"核销号 {writeoff_no} 不存在",
                writeoff_no=writeoff_no,
            )
        ])
    return _load_result(row, replayed=False)


def list_writeoffs(
    conn: sqlite3.Connection, reservation_no: str
) -> s.WriteoffListResponse:
    """列出某预留单的全部核销记录(按核销时间先后);预留不存在返回 404。"""
    reservation = conn.execute(
        "SELECT status FROM reservations WHERE reservation_no = ?",
        (reservation_no,),
    ).fetchone()
    if reservation is None:
        raise WriteoffError([
            _conflict(
                "reservation_not_found",
                f"预留号 {reservation_no} 不存在",
                reservation_no=reservation_no,
            )
        ])
    rows = conn.execute(
        "SELECT * FROM writeoffs WHERE reservation_no = ?"
        " ORDER BY writeoff_id",
        (reservation_no,),
    ).fetchall()
    return s.WriteoffListResponse(
        reservation_no=reservation_no,
        status=reservation["status"],
        writeoffs=[_load_result(r, replayed=False) for r in rows],
    )
