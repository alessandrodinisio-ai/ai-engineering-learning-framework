# Terminal & Shell

> The terminal is where AI engineers live. Get comfortable here.

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 1
**Time:** ~35 min

## Learning Objectives

- Filter and process training logs at the command line with pipes, redirection, and `grep`
- Create persistent tmux sessions with multiple panes to run training and GPU monitoring simultaneously
- Monitor system and GPU resources with `htop`, `nvtop`, and `nvidia-smi`
- Transfer files between local and remote machines with SSH, `scp`, and `rsync`

## The Problem

You'll spend more time in the terminal than in any editor. Training jobs, GPU monitoring, log tailing, remote SSH sessions, environment management. Every AI workflow touches the shell. If you're slow here, you're slow everywhere.

This lesson covers terminal skills that matter for AI work. No Unix history. No deep Bash scripting dives. Just what you need.

## The Concept

```mermaid
graph TD
    subgraph tmux["tmux session: training"]
        subgraph top["Top half"]
            P1["Pane 1: Training job<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["Pane 2: GPU monitoring<br/>watch -n1 nvidia-smi<br/>GPU: 78% | Mem: 14/24G"]
        end
        P3["Pane 3: Logs + experiments<br/>tail -f logs/train.log | grep loss"]
    end
```

Three things running at once. One terminal. You can detach, go home, SSH back in, and reattach. Training keeps running.

## Build It

### Step 1: Know Your Shell

Check which shell you're running:

```bash
echo $SHELL
```

Most systems use `bash` or `zsh`. Both work. This course's commands run in either.

Key things to know:

```bash
# Navigate
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# History search (most useful shortcut you'll learn)
# Ctrl+R then type part of a previous command
# Press Ctrl+R again to cycle through matches

# Clear screen
clear   # or Ctrl+L

# Cancel a running command
# Ctrl+C

# Suspend a running command (resume with fg)
# Ctrl+Z
```

### Step 2: Pipes and Redirection

Pipes chain commands together. This is how you process logs, filter output, and compose tools. You'll use it constantly.

```bash
# Count how many times "loss" appears in a log
cat train.log | grep "loss" | wc -l

# Extract only loss values from training output
grep "loss:" train.log | awk '{print $NF}' > losses.txt

# Watch a log file in real time, filtering for errors
tail -f train.log | grep --line-buffered "ERROR"

# Sort experiments by final accuracy
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

# Redirect stdout and stderr to different files
python train.py > output.log 2> errors.log

# Redirect both to the same file
python train.py > train_full.log 2>&1
```

Three redirections you need:

| Symbol | What it does |
|--------|-------------|
| `>` | Write stdout to file (overwrite) |
| `>>` | Append stdout to file |
| `2>` | Write stderr to file |
| `2>&1` | Send stderr to wherever stdout goes |
| `\|` | Pipe stdout of one command as stdin to the next |

### Step 3: Background Processes

Training jobs run for hours. You don't want a terminal open the whole time.

```bash
# Run in background (output still prints to terminal)
python train.py &

# Run in background, immune to hangup (closing terminal won't kill it)
nohup python train.py > train.log 2>&1 &

# See what's running in background
jobs
ps aux | grep train.py

# Bring a background job to foreground
fg %1

# Kill a background process
kill %1
# Or find its PID and kill
kill $(pgrep -f "train.py")
```

Difference between `&`, `nohup`, and `screen`/`tmux`:

| Method | Survives terminal close? | Can reattach? |
|--------|-------------------------|---------------|
| `command &` | No | No |
| `nohup command &` | Yes | No (check log file) |
| `screen` / `tmux` | Yes | Yes |

For anything longer than a few minutes, use tmux.

### Step 4: tmux

tmux lets you create persistent terminal sessions with multiple panes. It's the single most useful tool for managing training jobs.

```bash
# Install
# macOS
brew install tmux
# Ubuntu
sudo apt install tmux

# Start a named session
tmux new -s training

# Split horizontally
# Ctrl+B then "

# Split vertically
# Ctrl+B then %

# Move between panes
# Ctrl+B then arrow keys

# Detach (session keeps running)
# Ctrl+B then d

# Reattach
tmux attach -t training

# List sessions
tmux ls

# Kill a session
tmux kill-session -t training
```

A typical AI workflow session:

```bash
tmux new -s train

# Pane 1: start training
python train.py --epochs 100 --lr 1e-4

# Ctrl+B, " to split, then run GPU monitoring
watch -n1 nvidia-smi

# Ctrl+B, % to split vertically, tail logs
tail -f logs/experiment.log

# Now detach with Ctrl+B, d
# SSH out, grab coffee, come back
# tmux attach -t train
```

### Step 5: Monitoring with htop and nvtop

```bash
# System processes (better than top)
htop

# GPU processes (if you have an NVIDIA GPU)
# Install: sudo apt install nvtop (Ubuntu) or brew install nvtop (macOS)
nvtop

# Quick GPU check without nvtop
nvidia-smi

# Refresh GPU usage every second
watch -n1 nvidia-smi

# See which processes use GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

`htop` shortcuts you'll use:
- `F6` or `>` to sort by column (sort by memory to find leaks)
- `F5` to toggle tree view (see child processes)
- `F9` to kill a process
- `/` to search for a process name

### Step 6: SSH to Remote GPU Machines

When you rent a cloud GPU (Lambda, RunPod, Vast.ai), you connect via SSH.

```bash
# Basic connection
ssh user@gpu-box-ip

# With a specific key
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip

# Copy files to remote
scp model.pt user@gpu-box-ip:~/models/

# Copy files from remote
scp user@gpu-box-ip:~/results/metrics.json ./

# Sync an entire directory (faster for many files)
rsync -avz ./data/ user@gpu-box-ip:~/data/

# Port forwarding (access remote Jupyter/TensorBoard locally)
ssh -L 8888:localhost:8888 user@gpu-box-ip
# Now open localhost:8888 in your browser

# SSH config for convenience
# Add to ~/.ssh/config:
# Host gpu
#     HostName 192.168.1.100
#     User ubuntu
#     IdentityFile ~/.ssh/gpu_key
#
# Then just:
# ssh gpu
```

### Step 7: Useful Aliases for AI Work

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh
```

Or copy the ones you want. Key aliases:

```bash
# Quick GPU status
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# Kill all Python training processes
alias killtraining='pkill -f "python.*train"'

# Quick venv activation
alias ae='source .venv/bin/activate'

# Watch training loss
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

See `code/shell_aliases.sh` for the full set.

### Step 8: Common AI Terminal Patterns

These come up repeatedly in practice:

```bash
# Run training, log everything, notify when done
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "Training complete" you@email.com

# Compare two experiment logs side by side
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)

# Find largest model files (for disk cleanup)
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20

# Download a model from Hugging Face
wget https://huggingface.co/model/resolve/main/model.safetensors

# Unpack a dataset
tar xzf dataset.tar.gz -C ./data/

# Count lines in all Python files (see how big your project is)
find . -name "*.py" | xargs wc -l | tail -1

# Check disk space (training data fills disks fast)
df -h
du -sh ./data/*

# Check environment variables before training
env | grep -i cuda
env | grep -i torch
```

## Use It

When each tool comes up in this course:

| Tool | When you use it |
|------|----------------|
| tmux | Every training job (Phase 3 onward) |
| `tail -f` + `grep` | Monitoring training logs |
| `nohup` / `&` | Quick background tasks |
| `htop` / `nvtop` | Debugging slow training, OOM errors |
| SSH + `rsync` | Working on cloud GPUs |
| Pipes + redirection | Processing experiment results |
| Aliases | Saving time on repeated commands |

## Exercises

1. Install tmux, create a session with three panes, run `htop` in one, `watch -n1 date` in another, and a Python script in the third. Detach and reattach.
2. Add the aliases from `code/shell_aliases.sh` to your shell config and reload with `source ~/.zshrc` (or `~/.bashrc`).
3. Generate a fake training log with `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log`, then use `grep`, `tail`, and `awk` to extract only loss values.
4. Configure an SSH config entry for a server you have access to (or use `localhost` to practice the syntax).

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|----------------------|
| Shell | "the terminal" | The program interpreting your commands (bash, zsh, fish) |
| tmux | "terminal multiplexer" | A program that lets you run multiple terminal sessions in one window and detach/reattach |
| Pipe | "the bar thing" | The `\|` operator that sends one command's output as input to another |
| PID | "process ID" | A unique number assigned to each running process, used to monitor or kill it |
| nohup | "no hangup" | Runs a command immune to hangup signals, so closing the terminal won't kill it |
| SSH | "connect to server" | Secure Shell, an encrypted protocol for running commands on remote machines |
