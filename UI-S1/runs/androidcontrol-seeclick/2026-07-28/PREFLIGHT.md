# AndroidControl SeeClick Preflight

Status: `BLOCKED_NO_ANDROIDCONTROL_CHECKPOINT_OR_CONVERSION`

## Paper target

Values are `Type / Grounding / Step SR`.

| Setting | Anchor |
| --- | --- |
| Low | 93.0 / 73.4 / 75.0 |
| High | 82.9 / 62.9 / 59.1 |

These values are reported in the UI-TARS AndroidControl comparison table.

## Public SeeClick resources

- General grounding model: `cckevinn/SeeClick`
  - revision `b79618049920e209d15cda4d35004e32e8b485d9`
- AITW downstream model: `cckevinn/SeeClick-aitw`
  - revision `ad60143d8172108788b859b4d974087db1526808`
- Source: `njucckevin/SeeClick`
  - revision `0ef37ac4d7aaf37ba7b990e3e3c3ca77e1fb8f93`

No public checkpoint is labeled as AndroidControl-trained, and the SeeClick
release predates AndroidControl support. The AITW model uses SeeClick's AITW
prompt/action conversion; it is not automatically equivalent to the
AndroidControl Low/High unified-action setup.

## Resume condition

Before running the existing 7,708-step AndroidControl scorer, recover or verify:

1. which exact SeeClick checkpoint generated the reported AndroidControl rows;
2. the Low and High prompt/history conversion;
3. mapping from SeeClick/AITW actions into the OS-Atlas-style AndroidControl
   action schema and coordinate convention;
4. whether the result is zero-shot transfer or AndroidControl fine-tuning.

Using `SeeClick-aitw` with an invented conversion would produce a useful
diagnostic, but it must be labeled as a new transfer baseline rather than a
reproduction of the paper row.