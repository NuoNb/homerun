# Native backtest quote fidelity repair

Authorized by the owner on 2026-11-05 to repair the reproduced native backtest issues, review the changes, and commit after verification. Base commit: `81a4be5c`.

## Required behavior

- Historical discovery must use the latest actual state at or before each tick. Recorded empty books replace prior quotes and remain nontradable. One-sided books stay one-sided; future observations must not affect earlier ticks.
- Strategy SDK replay must expose actual recorded price/size levels, respect requested level limits, and use the same VWAP calculation as live book reads. Never put aggregate size at the best price or assume full fill probability without walking depth.
- Native exit decisions must receive executable best bid/ask, actual ladders, the snapshot timestamp, original entry signal time and strategy context. Signal time and fill time remain distinct. Replay clock and SDK reads during exits must use historical state and restore their previous contexts afterwards, including callback failures.
- The initial signal context belongs to an open position, survives partial fills and scale-ins, and is replaced on close/reopen. A callback must not mutate stored entry metadata accidentally.
- Recorded and projected events become visible at the first tick at or after observation; events beyond the last tick stay outside the replay. Persisted signals use payload emission time for window selection, ordering, entry time and resolution-window evaluation, with missing/null/empty timestamp fallback to row creation time.
- Equal positive submit/cancel p50 and p95 values mean constant latency, including 500, 1000 and 2000 milliseconds. Constant sampling must not consume random draws. Invalid negative, zero, reversed or non-finite quantiles remain rejected.
- Repair the shared infrastructure, without coupling tests to installed strategy slugs/classes. Retain real data and the frozen research protocol. Keep autoresearch disabled, holdout signal analysis sealed, and real trading off.

## Verification

Use regression tests that fail on the original implementation; run relevant existing backend suites inside Docker; repeat the saved real development-book contract probes. Review standards and specification separately before committing. Changes to unrelated Docker configuration and pre-existing research files are excluded from this commit.

## Verified result (2026-11-05)

- Relevant native backtest, market-data projection, replay SDK and strategy-backtester suites: 321 tests passed. Deployed-image focused regressions: 32 tests passed. Scoped Ruff checks and staged whitespace checks passed.
- PostgreSQL probes verified emission-time filtering/order, UTC offsets, missing/null/blank fallback and exclusive end boundaries. Resolution-window evaluation retained the correct 30 seconds instead of the row-creation-based 90 seconds.
- Saved real development Parquet probes passed: an empty BTC book replaces its prior quote; ETH bid 0.983 remains below the 0.99 exit threshold despite mid 0.9905; ask levels remain 13.3 at 0.998 and 35 at 0.999; constant 500/1000/2000 ms profiles are accepted.
- Independent standards and specification reviews each ended with zero unresolved findings. All findings raised during review were repaired and covered by regression checks.
- Only Homerun backend and jobs-worker containers were recreated. Both match all eight repaired source fingerprints; the other 13 container IDs are unchanged. The normal image build encountered an external Debian repository HTTP 502, so deployment reused the exact existing dependency image and overlaid the repaired sources.
- All five frozen research fingerprints are unchanged. Autoresearch remains at interval_seconds=0 and auto_apply=false. No candidate backtests, holdout signal analysis, shadow trading or real orders were started by this repair.

Detailed local evidence is retained under output/research/btc_eth_15m in native_repair_deployment_verified.json, native_repair_postgres_probe.json, native_repair_frozen_inputs_verified.json and round_001_native_contract_probe_after_repair.json.

## Second-review correction (2026-11-05)

The second independent review found two combined paths missing from the initial regression coverage: scanner catalogs could retain stale/synthetic prices through hydration and SDK fallback, and the WS-mid/resolution-time helpers could still read live state during replay. The initial 321 passing tests did not cover those combinations.

- Scanner projections now use the canonical Market shape with independent recorded token prices. Empty books replace the same market's catalog prices with unavailable values; hydration does not derive the other token by complement.
- Replay price, spread and bid/ask fill helpers read only installed historical books. Empty replay mappings remain isolated from live data; missing bid or ask is never synthesized. Both discovery dispatchers install an empty historical mapping when no books exist.
- Resolution time uses explicit now when supplied, then the replay clock, then the existing live wall clock.
- Five focused regressions failed against the previous deployed image. The repaired relevant suites, including input builders and crypto reconstruction, passed 341 tests. Additional permanent cases cover noncomplementary token prices, absent outcomes, bid-only, ask-only and entirely empty books. Both independent review findings were closed before remote publication.


## Native execution smoke correction (2026-11-05)

The first real development replay exposed a separate exit lifecycle defect: the engine submitted a complete exit again on each snapshot while earlier exits were pending venue admission. One bought position generated five full-size SELL fills, so that run is retained as invalid execution evidence and excluded from research results.

The engine now tracks active exit children per token, held side and strategy. Pending, working and partial children prevent another exit batch until every child is terminal. A rejected or cancelled IOC remainder releases the position for a retry sized from its actual remaining holdings. Single-order and laddered paths use the same tracking.

Three baseline regressions sold 100/200/400 shares against a 10-share position at 500/1000/2000 ms. The fix keeps the total at 10. Permanent cases also cover dense-snapshot laddered exits and partial IOC retries (10 shares close as 5 + 5). The focused engine, fidelity and settlement suites passed 80 tests. The frozen candidate, market inputs, fee model and holdout were unchanged.
