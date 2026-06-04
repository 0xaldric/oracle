# oracle

A computer-vision → on-chain **oracle**. It runs object detection on a live
video stream, turns the observed counts into a **commit-reveal attestation**,
signs it, and publishes it on-chain so anyone can verify the result.

## How it works
1. **Detect** — ingest a live stream and count objects with YOLOv8 + BoTSORT.
2. **Aggregate** — group detections into timed rounds (`round_manager`).
3. **Commit / reveal** — commit `keccak256(count, salt)`, then reveal and verify
   on-chain against the `DataAttestation` contract (`signer.py`).

## Stack
Python · YOLOv8 · web3 / eth-account · Node publisher

## Ops
Runs on a GPU server as long-lived `systemd` services with a watchdog for
recovery across rounds. See `DEPLOY.md` for the operations guide.
