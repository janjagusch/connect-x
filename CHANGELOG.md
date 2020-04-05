# Changelog

## Unreleased Changes

- Added `docs/DEVELOPMENT.md` to facilitate onboarding of new developers.
- Made `.mark` attribute in `connect_x.game.connect_x.ConnectXState` protected.

## 0.7.0

- Added benchmarking environment in `/benchmark`.
- Fixed logging mistake in `submission.py`.
- Added processing time benchmarking tool in `/notebooks`.
- Added iterative deepening.

## 0.6.0

- The project now represents boards as bitmaps and uses caching during the Minimax algorithm.

## 0.5.0

- The project now works completely mark independent and matrix based.
- The precomputed best actions now have a depth of 8.

## 0.4.1

- Fixed wrong file name in `.travis.yml`.

## 0.4.0

- Added `connect_x/board_action_map.py`, which provided access to precomputed best actions for boards in the first N turns.

## 0.3.0

- Set `max_depth=3` in `submission.py`.
- Added `connect_x.move_catalogue` to copy with the dummy trick at the beginning of the game.

## 0.2.0

- Updated instructions in `README.md` on how to submit through Travis CI.
- Added Minimax algorithm.

## 0.1.3

- Fixed and cleaned up `.travis.yml` and `bin/` scripts.

## 0.1.2

- Fixed `bin/check_release` script again.

## 0.1.1

- Fixed `bin/check_release` script.

## 0.1.0
- Initialized `connect-x`.

