@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0.."

echo.
echo ============================================================
echo  Beer Game MARL -- 15k Episode Learning Validation Run
echo ============================================================
echo  Each run uses:
echo    warm_up_episodes : 1000  (fills buffer / random rollout)
echo    total_episodes   : 15000 (14000 actual training episodes)
echo    patience         : 2000  (early stop if plateau confirmed)
echo    epsilon_decay    : 5000  (epsilon 1.0 -> 0.05 over 5k eps)
echo    lr_scheduler     : step 2000 / gamma 0.5
echo    seed             : 42
echo ============================================================
echo.

REM ---------------------------------------------------------------
REM 1/5  IPPO
REM      Independent PPO -- each agent uses its own local obs/critic
REM ---------------------------------------------------------------
echo [1/5] IPPO
python scripts/train_ppo.py ^
    agent=ippo ^
    total_episodes=15000 ^
    seed=42 ^
    hydra.run.dir=outputs/ippo
if %errorlevel% neq 0 (
    echo IPPO run FAILED with exit code %errorlevel%
    pause
    exit /b %errorlevel%
)
echo IPPO done.
echo.

REM ---------------------------------------------------------------
REM 2/5  MAPPO
REM      Centralised critic on global state, shared actor
REM ---------------------------------------------------------------
echo [2/5] MAPPO
python scripts/train_ppo.py ^
    agent=mappo ^
    total_episodes=15000 ^
    seed=42 ^
    hydra.run.dir=outputs/mappo
if %errorlevel% neq 0 (
    echo MAPPO run FAILED with exit code %errorlevel%
    pause
    exit /b %errorlevel%
)
echo MAPPO done.
echo.

REM ---------------------------------------------------------------
REM 3/5  Comm-MAPPO
REM      MAPPO + Gumbel-softmax communication channel
REM ---------------------------------------------------------------
echo [3/5] Comm-MAPPO
python scripts/train_ppo.py ^
    agent=comm_mappo ^
    total_episodes=15000 ^
    seed=42 ^
    hydra.run.dir=outputs/comm_mappo
if %errorlevel% neq 0 (
    echo Comm-MAPPO run FAILED with exit code %errorlevel%
    pause
    exit /b %errorlevel%
)
echo Comm-MAPPO done.
echo.

REM ---------------------------------------------------------------
REM 4/5  QMIX
REM      Per-agent GRU Q-nets + monotonic mixing network
REM      Trains from ep 33 (buffer > batch_size=32), no warm_up guard
REM ---------------------------------------------------------------
echo [4/5] QMIX
python scripts/train_qmix.py ^
    agent=qmix ^
    total_episodes=15000 ^
    seed=42 ^
    hydra.run.dir=outputs/qmix
if %errorlevel% neq 0 (
    echo QMIX run FAILED with exit code %errorlevel%
    pause
    exit /b %errorlevel%
)
echo QMIX done.
echo.

REM ---------------------------------------------------------------
REM 5/5  Comm-QMIX
REM      QMIX + differentiable latent communication channel
REM      warm_up=1000 so training starts at ep 1000
REM ---------------------------------------------------------------
echo [5/5] Comm-QMIX
python scripts/train_comm_qmix.py ^
    agent=comm_qmix ^
    total_episodes=15000 ^
    seed=42 ^
    hydra.run.dir=outputs/comm_qmix
if %errorlevel% neq 0 (
    echo Comm-QMIX run FAILED with exit code %errorlevel%
    pause
    exit /b %errorlevel%
)
echo Comm-QMIX done.
echo.

echo ============================================================
echo  All 5 runs complete. Check W^&B project BeerGame_Research.
echo  Best weights saved under weights_ippo/, weights_mappo/,
echo  weights_comm_mappo/, weights_qmix/, weights_comm_qmix/
echo ============================================================
pause
