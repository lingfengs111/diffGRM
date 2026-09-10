# Machine-owned incoming records

Each producer writes only below `incoming/<machine-id>/`. Configure a stable
identifier once per clone with:

```bash
git config --local diffgrm.machine gpu-ckpt
```

Use names such as `primary`, `gpu-ckpt`, and `gpu-git`. A machine may maintain
its own mutable `STATUS.md` beside its immutable record directories. Different
machines must never edit one another's incoming directories.
