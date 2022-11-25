#!/bin/bash

python tools/test.py configs/raft/raft_8x2_50k_wirn22.py \
$MMFLOW/work_dir/raft_dCADDY_freezed_07/iter_40000.pth \
--eval EPE
