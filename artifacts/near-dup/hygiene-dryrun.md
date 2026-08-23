# Near-duplicate hygiene: dry-run report

Produced 2026-08-23 by `scripts/lm_collapse_near_dups.py` against a **copy** of the
live Living Memory database. Nothing was applied; this report is the input the
operator needs in order to decide whether to apply, and the exact command to do it.

## How this was produced, and what was never touched

The live database `~/.local/share/living-memory/global.sqlite3` (648 MB plus a
503 MB WAL, a server writing to it the whole time) was only ever opened read-only.
The copy was taken from inside python, because there is no `sqlite3` CLI on this
host and a plain `cp` of a WAL database copies half a database:

```python
import sqlite3
con = sqlite3.connect("file:/home/sfx/.local/share/living-memory/global.sqlite3?mode=ro", uri=True)
con.execute("VACUUM INTO ?", ("/tmp/lm-neardup-copy.sqlite3",))   # 2.1 s, 656 MB
```

`VACUUM INTO` writes a new file and reads the source through a `mode=ro` connection,
so the WAL is folded in and the source is not written. The copy carries the same
13,442 active nodes and 63,179 chunk rows as the source.

A second dry run was pointed straight at the live file, again read-only, to confirm
the script behaves the same on a database with a hot WAL under a running server: it
reported 13,443 active nodes (the server had added one), the same 1,472 pairs and
the same 125 collapses. `--dry-run` is the default and opens the database with
`file:...?mode=ro`, so a dry run cannot write even by accident; the only INSERT in
the script lives behind `if args.apply`.

## The command for the operator

Read the distribution below first. Then, **with the server that owns this database
stopped** (or restarting it straight afterwards):

```bash
# 1. look again, on the live file, read-only -- this writes nothing
python3 scripts/lm_collapse_near_dups.py \
    /home/sfx/.local/share/living-memory/global.sqlite3 --threshold 0.95

# 2. stop the server that owns this database (operator step, not the agent's)

# 3. apply -- writes 125 supersedes edges, deletes nothing
python3 scripts/lm_collapse_near_dups.py /home/sfx/.local/share/living-memory/global.sqlite3 --threshold 0.95 --apply

# 4. start the server again
```

The apply run prints this note before and after writing:

```
Run this against a database whose server is STOPPED, or restart that server
immediately afterwards. A running server keeps the scope chunk matrix cached
in memory; collapsing nodes underneath it leaves that cache stale, so recall
keeps serving the nodes this pass just superseded until the process restarts.
The edges written here also bypass the server's own supersedes hook, so query
anchor edges still point at superseded nodes until the next anchor maintenance
pass; recall demotes superseded nodes regardless, so this is a ranking nicety,
not a correctness hole.
```

## Preflight

```
## preflight
database                : /tmp/lm-neardup-copy.sqlite3
opened                  : read-only (dry run)
numpy                   : 2.4.3
node_chunk_embeddings   : present, 62921 rows for active nodes
vector width            : 384 (uniform)
chunk coverage          : 13428 of 13442 active nodes (99.9%), floor 50.0%
connections table       : present
```

## Distribution FIRST: max cosine per node, on THIS database

This is printed before the threshold is applied, and it is the reason the threshold
is an argument rather than a constant.

It also reproduces the calibration this whole goal was built on. That calibration
(median max-cosine 0.829, 550 nodes with a neighbour at >= 0.95, 203 at >= 0.99) was
measured over **traces only, across all scopes**. Restricting this script's scan the
same way returns median **0.8289**, **550** nodes at >= 0.95 and **203** at >= 0.99 --
the same three numbers, computed by a script that shares no code with the system that
produced them. The numbers below are higher (median 0.8321, 991 nodes at >= 0.95)
because they include concepts and schemas, which are the most duplicated levels in
this corpus:

| slice | nodes | median max-cos | >= 0.95 | >= 0.99 |
| --- | --- | --- | --- | --- |
| traces only, cross-scope (the goal's calibration) | 12,881 | 0.8289 | 550 (4.3%) | 203 |
| traces only, same-scope | 12,868 | 0.8218 | 519 (4.0%) | 195 |
| all levels, same-scope (**this report**) | 13,415 | 0.8321 | 991 (7.4%) | 236 |
| all levels, cross-scope | 13,428 | 0.8382 | 1,026 (7.6%) | 246 |

```
## max-cosine distribution for THIS database
Calibration does not transfer between corpora. These are the numbers of
the database named above, measured before any threshold was applied.

nodes compared          : 13415
pairing                 : within the same scope
pairs compared          : 16861028
nodes without chunks    : 14 (excluded: nothing to compare)
nodes alone in their scope: 13 (no in-scope neighbour to compare with)

max-cosine percentiles (per node, against its nearest other node):
  mean   0.8252
  min    0.4414
  p10    0.7071
  p25    0.7626
  p50    0.8321
  p75    0.8906
  p90    0.9371
  p95    0.9631
  p99    0.9993
  max    1.0000

counts at meaningful thresholds:
  cosine         nodes share of nodes      pairs
  >= 0.800        8295         61.83%     115723
  >= 0.850        5739         42.78%      25498
  >= 0.900        2845         21.21%       6607
  >= 0.925        1753         13.07%       3560
  >= 0.950         991          7.39%       1472
  >= 0.970         547          4.08%        519
  >= 0.990         236          1.76%        231
  >= 0.999         141          1.05%        132

scan took 1.6 s (numpy)
```

## Threshold, and the counts at it

```
## threshold
threshold applied       : 0.9500
band policy             : >= 0.95 collapsible; 0.85-0.95 left alone by default (different facts, and concept/source-trace provenance)
length guard            : candidate may not exceed bearer by more than 10%
provenance guard        : never supersede a node listed in source_traces of a live concept/schema node

provenance-protected nodes: 6041 of 13442 active (source_traces of live concept/schema)
existing supersedes edges : 491

## pairs at or above the threshold
pairs                   : 1472
would collapse          : 125
skipped                 : 1347
```

At 0.95, **1,472 pairs** are near-duplicates and **125 collapse**: 0.93% of the
13,442 active nodes and 3.70% of the live text (572,930 of 15,477,396 chars).
The lowest collapsed pair sits at cosine 0.9502; 35 of the 125 are `schema` nodes,
82 `trace`, 8 `concept`.

## What would be collapsed (all 125)

`supersede` gets a `supersedes` edge from `keep`; the row itself stays where it is.

| cosine | supersede | keep | level | scope | chars |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | `01KY5HQA417E6882ZRBVZD0ATP` | `01KRVVE5XQ9CJ7JPVPWFW9102F` | schema | project:ae | 3650 -> 3650 |
| 1.0000 | `01KZ8RYAPF3JZ6RN65FFRG9WR8` | `01KRVVE5XQ9CJ7JPVPWFW9102F` | schema | project:ae | 3650 -> 3650 |
| 1.0000 | `01KZEXTZYEXQVQTQMD9DQRJZBB` | `01KRVVE5XQ9CJ7JPVPWFW9102F` | schema | project:ae | 3650 -> 3650 |
| 1.0000 | `01KZZP9QFWVKZWP2KVK55PNRPN` | `01KRVVE5XQ9CJ7JPVPWFW9102F` | schema | project:ae | 3650 -> 3650 |
| 1.0000 | `01M0FAA18EPMCPF2PSNMK2Y81S` | `01KRVVE5XQ9CJ7JPVPWFW9102F` | schema | project:ae | 3650 -> 3650 |
| 1.0000 | `01KY5HQA2V9M9VRY3V7HXT4Z6Z` | `01KRWVEF08GR0365E296ZDNDWH` | schema | project:ae | 3700 -> 3700 |
| 1.0000 | `01KZ8RYAN6A419XNF07RTDG2WM` | `01KRWVEF08GR0365E296ZDNDWH` | schema | project:ae | 3700 -> 3700 |
| 1.0000 | `01KZEXTZXDHSHCMQ2Z8Q3CC28Z` | `01KRWVEF08GR0365E296ZDNDWH` | schema | project:ae | 3700 -> 3700 |
| 1.0000 | `01KZZP9QEPF01F8CCZZQ2J79XQ` | `01KRWVEF08GR0365E296ZDNDWH` | schema | project:ae | 3700 -> 3700 |
| 1.0000 | `01M0FAA1814NGB93T61EAECX23` | `01KRWVEF08GR0365E296ZDNDWH` | schema | project:ae | 3700 -> 3700 |
| 1.0000 | `01KY5HQ9QYAPMMTAV8T9074VRS` | `01KT7QN1JK2CAWM34QADEA3YCP` | schema | project:ae | 22660 -> 22660 |
| 1.0000 | `01KZ8RYAC3N836NFZ7DJBS03HM` | `01KT7QN1JK2CAWM34QADEA3YCP` | schema | project:ae | 22660 -> 22660 |
| 1.0000 | `01KZEXTZJV576E1PH4J9ZQN8B4` | `01KT7QN1JK2CAWM34QADEA3YCP` | schema | project:ae | 22660 -> 22660 |
| 1.0000 | `01KZZP9Q3YB1DZT0FHGZY17QSN` | `01KT7QN1JK2CAWM34QADEA3YCP` | schema | project:ae | 22660 -> 22660 |
| 1.0000 | `01M0FAA12QJEG9181F01BX7M0P` | `01KT7QN1JK2CAWM34QADEA3YCP` | schema | project:ae | 22660 -> 22660 |
| 1.0000 | `01M0FBSBYAN6BRMRWH67C2HSE4` | `01KT7QN1JK2CAWM34QADEA3YCP` | schema | project:ae | 22660 -> 22660 |
| 1.0000 | `01KY5HQ9VB4Q89YNA0GAVT6V89` | `01KT7QN1M6RD8NK4RZB5N5X9MR` | schema | project:ae | 32797 -> 32797 |
| 1.0000 | `01KZ8RYAFD04XJJFE50HHGSJHK` | `01KT7QN1M6RD8NK4RZB5N5X9MR` | schema | project:ae | 32797 -> 32797 |
| 1.0000 | `01KZEXTZP52XF5ZFT2EA7WQDSQ` | `01KT7QN1M6RD8NK4RZB5N5X9MR` | schema | project:ae | 32797 -> 32797 |
| 1.0000 | `01KZZP9Q7PSB52RCCK8A353MKR` | `01KT7QN1M6RD8NK4RZB5N5X9MR` | schema | project:ae | 32797 -> 32797 |
| 1.0000 | `01M0FAA14G62BRDMTYQZDEMBNR` | `01KT7QN1M6RD8NK4RZB5N5X9MR` | schema | project:ae | 32797 -> 32797 |
| 1.0000 | `01M0FBSBZYD124QFSGDY2EQPP8` | `01KT7QN1M6RD8NK4RZB5N5X9MR` | schema | project:ae | 32797 -> 32797 |
| 1.0000 | `01KWHYQJZQBBAMN234WKXD4JJG` | `01KVC1S19AN2KEJ851B5BZTYT6` | schema | project:x | 2631 -> 2631 |
| 0.9993 | `01KSQ335AEA39CEWX5HHMVW0Q0` | `01KSMXRB8VZXD89HJCAY996JEK` | schema | project:octopus | 19768 -> 18695 |
| 0.9976 | `01KSAZD11GAJ1PX2C9AVPGQ1GA` | `01KSAZD1N0HKFG9TE4YQ78CAYJ` | trace | project:octopus | 285 -> 285 |
| 0.9945 | `01KSR8A5P3D6C2FEBME1MHXWT5` | `01KS81EPTT5AVNHRMA933ENKBC` | trace | project:online | 158 -> 147 |
| 0.9937 | `01KS18Y9GFY5N4ZPYDHJCZ71PR` | `01KRYA6QAC7KAA3SP1N1JZJJMT` | trace | project:octopus | 160 -> 155 |
| 0.9935 | `01KTRR6WHFXFW16M691Q1N98E1` | `01KTRJCB5FQ7ZT2WKD02TQ20B3` | trace | project:x | 220 -> 211 |
| 0.9931 | `01KRX759ASWP7GKSG4NX3K0284` | `01KRVQNC96TW2K70A3JENYZXMP` | trace | project:ae | 150 -> 147 |
| 0.9931 | `01KSB3ZT0S8E97Q53TH3TT6DR6` | `01KSB3T1DJRNS2SMCR9189FBZS` | trace | global | 81 -> 78 |
| 0.9920 | `01KSFNCKH3Y443YS9CFCXPK5PC` | `01KSMXRB8VZXD89HJCAY996JEK` | schema | project:octopus | 14568 -> 18695 |
| 0.9906 | `01KVRQ3FPY8R649R1VG984KA5K` | `01KVRKQM12J1E3QZB1DM7550TX` | trace | project:x | 338 -> 350 |
| 0.9901 | `01KVRQAP2ADN6YWCRVKKYMMDC3` | `01KVRJZ9TYQ4VBTAMN19FE6TFK` | trace | project:x | 352 -> 332 |
| 0.9899 | `01KRRPZ1Y81TDQBB2B5M5FM2YH` | `01KRT6P4EPHQS2DD2SWQXPZT0A` | trace | project:ae | 2252 -> 2326 |
| 0.9897 | `01KSQK8ECMZPQQS28HWR3P14BT` | `01KSQRZ18QF1AY3TE4FKSGZJRZ` | trace | project:online | 339 -> 373 |
| 0.9883 | `01KZYJR01MQMFGMRZEMMQGRXBC` | `01KZW430B4PPQPJ2ZG4AC7DG5Z` | concept | project:lm | 1195 -> 1191 |
| 0.9871 | `01KZW4308WTF6ZRMJ9ZRY7G6ED` | `01KZYRW9VJQHBS90MWBY8AQ58P` | concept | project:lm | 1180 -> 1180 |
| 0.9864 | `01M0Q2JG25A59JN0Z7WC3ND791` | `01M0FKQ19SXWTJG7ZMQWZQ4PGR` | concept | project:living-memory | 1163 -> 1163 |
| 0.9863 | `01KRS10087DBPPZ6DG0GT0E60E` | `01KRS0ZYEWWNVNN70MSRRCZWYC` | trace | project:octopus | 2804 -> 3762 |
| 0.9860 | `01KRXJKFDR6H2JSYR2NBZEMNR7` | `01KRXG9E6GYN291EQJGWSB2RNF` | trace | project:ae | 147 -> 163 |
| 0.9857 | `01KRXJB5BAG6N6HKJDXVWN12G1` | `01KRXG9E6GYN291EQJGWSB2RNF` | trace | project:ae | 155 -> 163 |
| 0.9857 | `01KS7X135RRBYH3HZJG7C23CV9` | `01KS80ZEFV23M0RCGCH8ZHSHPC` | trace | project:online | 399 -> 399 |
| 0.9850 | `01KS97R8GQXW3EC48PWETVKCNS` | `01KS93EKA45V7BKP7JBB034K4B` | trace | project:octopus | 1934 -> 2041 |
| 0.9842 | `01KT9ZPG3AT3KD63NTEVFT2BWC` | `01KT9SGNF0A9TS70NJK5K7WC3Z` | trace | project:online | 529 -> 516 |
| 0.9836 | `01KRWVEF09CJ2N7F7E1YKVWAR3` | `01KRVDY1Q1NTDHFV9S5G9D9VW7` | schema | project:ae | 2044 -> 9024 |
| 0.9829 | `01KSRQ2YS0PMYKP3C1QBPNY7NE` | `01KSQRZ18QF1AY3TE4FKSGZJRZ` | trace | project:online | 358 -> 373 |
| 0.9818 | `01KRP2RBD1AQ6RC4QK7XASYBST` | `01KRNWYMPWFHK9C2031EBJKW3H` | trace | project:online | 403 -> 403 |
| 0.9816 | `01KSR5FCJ9MK1AM94704ZJM7JX` | `01KSQ1DFNZ83GSGDKG4JVA4415` | trace | project:octopus | 155 -> 187 |
| 0.9816 | `01KT9T6BBQYVZN5DW5XJYWD3C8` | `01KT9SGNF0A9TS70NJK5K7WC3Z` | trace | project:online | 540 -> 516 |
| 0.9808 | `01KVRMA23KK80DMTVT11JWTQH3` | `01KVRJ23147XQRAW8PBE6HMT7C` | trace | project:x | 327 -> 325 |
| 0.9807 | `01KSQWRG4ET22RZTC2TKSG0T11` | `01KSQRZ18QF1AY3TE4FKSGZJRZ` | trace | project:online | 408 -> 373 |
| 0.9806 | `01KS7WV2Q3XWTYQB1TMMY0KMXC` | `01KS7W7T241MMGFJPP9BVE1K74` | trace | project:lm | 125 -> 148 |
| 0.9804 | `01KRXJWQV9BJ56Z4GQSWB6B19H` | `01KRVQNC96TW2K70A3JENYZXMP` | trace | project:ae | 150 -> 147 |
| 0.9796 | `01KRS0ZZ35Z11X5EWZ2KVE0GZR` | `01KRS0ZYEWWNVNN70MSRRCZWYC` | trace | project:octopus | 3771 -> 3762 |
| 0.9779 | `01KV1KRH09QCRAQXWP6CVVHQT6` | `01KV1JWWXN9PDVWK14JBVB3TET` | trace | project:x | 188 -> 192 |
| 0.9777 | `01KRW0A276M9TBN0SVYMD3S0ZQ` | `01KRW09XRXC9Z15M826C9DZFR1` | trace | project:ae | 209 -> 214 |
| 0.9768 | `01KS9R3QJA8JBJ7QFEBWDTABX8` | `01KS9R2PGCX3PK1Z3Z34NSYG0E` | trace | project:octopus | 308 -> 308 |
| 0.9768 | `01KS1BK2H7YT02PS8KTR6YZH2S` | `01KS1AB6RPV3JBKADJFZFQ32ZW` | trace | project:octopus | 999 -> 1023 |
| 0.9765 | `01KT9VDSW0YEFZKQFJDNEQA1MJ` | `01KT9SGNF0A9TS70NJK5K7WC3Z` | trace | project:online | 540 -> 516 |
| 0.9754 | `01KSRV8SQ69PGJ296XW4F16HMZ` | `01KSQRZ18QF1AY3TE4FKSGZJRZ` | trace | project:online | 374 -> 373 |
| 0.9727 | `01KV5BP2ENKPSRK9GME40S2CMX` | `01KV4B73BJ7HBSCRM3QWHKDYTM` | trace | project:x | 518 -> 509 |
| 0.9710 | `01KTV02NVWMZF49M3TF3XD104Z` | `01KT3THPSVF31R7MET6KJTVA8H` | trace | project:online | 381 -> 381 |
| 0.9707 | `01KS9AJF5EXN200JQC7MV8BASS` | `01KS95H5T1D5VD2GZ47GBFF49S` | trace | project:octopus | 2190 -> 2044 |
| 0.9706 | `01KVRP0W933R3C4ZBK4VYT0EVA` | `01KVRJKRB5XJJJD0J00RMT9EQR` | trace | project:x | 344 -> 335 |
| 0.9700 | `01KRXZS1MJYZYVZ0H8JMRNG2QN` | `01KRWHQTBDBGQC6WJ1YGYTY6YT` | trace | project:ae | 170 -> 162 |
| 0.9699 | `01KT9W9R2C1ZTH0TS2JJGAJDWH` | `01KT9S3S7TAXRED02Y40VGX2EV` | trace | project:online | 531 -> 553 |
| 0.9691 | `01KRZ8EKDXN4H7JZCYFMFG8J6M` | `01KRVAJ5QMT3HHS08DF7P17PVM` | trace | project:ae | 179 -> 177 |
| 0.9686 | `01KS962WHBDTE6X9QRZJ81P6D5` | `01KS95H5T1D5VD2GZ47GBFF49S` | trace | project:octopus | 2070 -> 2044 |
| 0.9679 | `01KVRPGQHBEBANSCG0F8R5RBP0` | `01KVRJ23147XQRAW8PBE6HMT7C` | trace | project:x | 322 -> 325 |
| 0.9674 | `01KTA24P3NVX47HXCS2269PQR1` | `01KT9SGNF0A9TS70NJK5K7WC3Z` | trace | project:online | 498 -> 516 |
| 0.9670 | `01KS97Q68YZKHZ3T1NY27XKBH1` | `01KS95H5T1D5VD2GZ47GBFF49S` | trace | project:octopus | 1898 -> 2044 |
| 0.9659 | `01KZYJFS26HV0NGHAEXFZP42T3` | `01KZYJR05YBA7DB9TM2MH49AER` | concept | project:lm | 1192 -> 1111 |
| 0.9654 | `01KS98AA0M2Q25C4102M4TE59V` | `01KS9A9MXW40DM7NSS43X0JZWN` | trace | project:octopus | 1971 -> 1837 |
| 0.9653 | `01KSRE9QW0JZ1MVSZC72ACYB2H` | `01KSRCNV043HHTY3JGVYMDW8TJ` | trace | project:online | 148 -> 167 |
| 0.9647 | `01KS8PVNM3N2YA862ZXQ8G1QSB` | `01KS8PWBW0TQWT2BRG5FTQ86PH` | trace | project:octopus | 185 -> 171 |
| 0.9635 | `01KS9AYJP06KVE4X4BJ9N8RKBF` | `01KS98J9FFB1GNKC7R22DYKDS3` | trace | project:octopus | 2263 -> 2358 |
| 0.9632 | `01KT9WKGW680HRF1CY3NJ4WYWX` | `01KT9S3S7TAXRED02Y40VGX2EV` | trace | project:online | 583 -> 553 |
| 0.9624 | `01KRS1GA0APCM203G6AFGH37D8` | `01KRS1G6DGHCVTR1N2JCHCEDBS` | trace | project:octopus | 5278 -> 64472 |
| 0.9620 | `01KRS0ZZJR11FHSG7YSC3TTJ23` | `01KRS0ZYEWWNVNN70MSRRCZWYC` | trace | project:octopus | 3784 -> 3762 |
| 0.9617 | `01KRX4WCDG1YCWH2XF6SMGD7DX` | `01KRWHQKPCX9ZYHFZYVWCYJT5Z` | trace | project:ae | 144 -> 142 |
| 0.9615 | `01KVRMSQ6M920CAZGTW2MS1YWG` | `01KVRJHA2QYPX56APZMS9R21TR` | trace | project:x | 335 -> 318 |
| 0.9613 | `01KTSB8N2E6HA4CSF2DXBQYK80` | `01KTCH9T0TJJV17H9K3KB6WSV1` | schema | project:online | 1241 -> 1161 |
| 0.9610 | `01KRTS51PQCC82F0ADC83ZSYAN` | `01KRTS512MXHCQTFMKKS2XEP3A` | concept | project:lm | 214 -> 210 |
| 0.9608 | `01KSMSAPP3JJAV82729M17EV0V` | `01KSMR8833BF6Q714W752YPHVA` | trace | project:octopus | 1135 -> 1283 |
| 0.9604 | `01KVRN4A6AFYBF3GQQP0TNY3KT` | `01KVRJZ9TYQ4VBTAMN19FE6TFK` | trace | project:x | 354 -> 332 |
| 0.9602 | `01KTNGMYMNYS1RB8EKX5HHBDAB` | `01KTCH9T0TJJV17H9K3KB6WSV1` | schema | project:online | 1274 -> 1161 |
| 0.9599 | `01KS5HMWJJ237D6FFCEPWVFJHB` | `01KS5JZ6VQNB1GDGPY8TR65F28` | trace | project:online | 558 -> 566 |
| 0.9597 | `01KS9BE9XS1SBPYSQR8MFP8Q73` | `01KS969ESDE1D0Q5HBHT0BDFM2` | trace | project:octopus | 564 -> 557 |
| 0.9591 | `01KRTS51PRKS1YEJ29NP6T96GZ` | `01KRTS51DCF505AXAK7RMNWKT2` | concept | project:lm | 266 -> 260 |
| 0.9580 | `01KRS1G0R59DE97ZDY4J1W72V3` | `01KRS1G6DGHCVTR1N2JCHCEDBS` | trace | project:octopus | 57088 -> 64472 |
| 0.9578 | `01KTSKWPGKT0WDQWXE6J8DWXFS` | `01KRXY78C47DT7F4Z1M4DRWEVZ` | schema | project:octopus | 1499 -> 4126 |
| 0.9577 | `01KZW430ENB75WE2FYSSFFTQ1A` | `01KZYRW9YTA5YG8WAQ5VPVMK66` | concept | project:lm | 1156 -> 1156 |
| 0.9572 | `01KVHDAAX6B1MHJNERES076V3Q` | `01KVHDM4T4GESY5YR5W7W8WXY1` | trace | project:x | 2682 -> 2950 |
| 0.9572 | `01KRS10A6BJPXWXXYDXJQQMKQ3` | `01KRS109KSGEDX2DCCC2AFGC3K` | trace | project:octopus | 3730 -> 3683 |
| 0.9569 | `01KWWHB79A84X915J0YR7WNKEY` | `01KZ8S65TXD79TJE7XV056Y2EE` | trace | project:ae | 2227 -> 2248 |
| 0.9568 | `01KTA5KTCP2B7Y103W9NP53H1D` | `01KTA3NCKN4GJYEME42FBZSPE3` | trace | project:online | 116 -> 158 |
| 0.9567 | `01KS5FWP7E5EC0YCCFSDJMA3GF` | `01KS5EYX0ADB13Q5A1CKQGS9G6` | trace | project:online | 520 -> 561 |
| 0.9561 | `01KSRC3KY2J9TYG2SEGAZG3HV8` | `01KSQ1DFNZ83GSGDKG4JVA4415` | trace | project:octopus | 201 -> 187 |
| 0.9558 | `01KVDSMECAM29A7PDX8GZ6H6J6` | `01KVDKN3FQ50EFC30QSEMDV77N` | trace | project:x | 1641 -> 1880 |
| 0.9557 | `01KT9W4BK1JKTWK2KY6W6D8JZ8` | `01KT9S3S7TAXRED02Y40VGX2EV` | trace | project:online | 590 -> 553 |
| 0.9549 | `01KS17D4E1NVF6QMJ3JJZE6JEQ` | `01KS0T38SEE6713ZWN0JA45C47` | trace | project:ae | 148 -> 161 |
| 0.9549 | `01KW6X7SC2TGJDGDV4M8G61ANG` | `01KW6WE0WDRXF17N1R9WRNXGW8` | trace | project:x | 4134 -> 4788 |
| 0.9547 | `01KT9WF6ST66BMN50R1JVES3K4` | `01KT9S3S7TAXRED02Y40VGX2EV` | trace | project:online | 493 -> 553 |
| 0.9545 | `01KVYS3SMZ01CZBDX04QJ0NS4T` | `01KTCH9T0TJJV17H9K3KB6WSV1` | schema | project:online | 1206 -> 1161 |
| 0.9544 | `01KVRSYKKJ2R02EC5BTSEF7NT6` | `01KVRSYKJYA1C8YD9K4F59J8XS` | schema | project:x | 2144 -> 5129 |
| 0.9540 | `01KX374MMT5N096HC3CQAWCX02` | `01KX34DMJECNK45KC1F8XY2836` | trace | project:online | 162 -> 162 |
| 0.9540 | `01KT873561Z5ZY1NE8R6VJBRPK` | `01KT7C5RKYQRGRYE8FR4ZAWCV1` | schema | project:octopus | 2055 -> 5280 |
| 0.9539 | `01KRS128TH842BV8BXA0ZGAA79` | `01KRS12HZEZ2B4V5FNYZDS7C65` | trace | project:octopus | 3738 -> 3702 |
| 0.9538 | `01KV2BQR4GB87MF0G0YS4ZWDQ0` | `01KV2947ZM9EPRHTQZ7Z55DP48` | trace | project:x | 212 -> 212 |
| 0.9534 | `01KRTS51PNR91GP1K8M329DYJY` | `01KRTS512MXHCQTFMKKS2XEP3A` | concept | project:lm | 212 -> 210 |
| 0.9531 | `01KRS1EMMJEF0X3VXW1PETNWT1` | `01KRS1E3XX8H48ZJ9678EBR849` | trace | project:octopus | 2232 -> 5005 |
| 0.9525 | `01KS0QKC6GA350T4HKWM0R4BNS` | `01KS0XVB1RPVS27HJZ8ZGS0ES6` | trace | project:octopus | 620 -> 583 |
| 0.9524 | `01KVFRV26V42KX58EF6B9PCNFJ` | `01KV68151P7F05H6NVR85BPQEH` | schema | project:x | 2530 -> 2692 |
| 0.9524 | `01KT7C5RKH6QAV6Z8SB19G5QDP` | `01KT4WNA0M9Q2Z7111XWQ8EFV9` | schema | project:octopus | 4955 -> 4600 |
| 0.9520 | `01KTBXTAP97M36K2MAKFW5DSK9` | `01KTBYKNVF4YWZED016PH7T1VT` | trace | project:x | 1435 -> 1405 |
| 0.9519 | `01KSQ3BX7VD6FY7K55BYJGM7QQ` | `01KSQG2M6PJ3KQSN67PMBSQ46D` | trace | project:octopus | 2205 -> 4048 |
| 0.9516 | `01KRYC860HQVM92Y5KZ3RQ09FN` | `01KRYAJ4M7M3S8YXCGJNBA0CJM` | trace | project:octopus | 988 -> 1188 |
| 0.9515 | `01KS1KFEBXREX4FXPN4NR1ZYH8` | `01KRYA6QAC7KAA3SP1N1JZJJMT` | trace | project:octopus | 158 -> 155 |
| 0.9514 | `01KSRAH8J0DR0XC485SYDBWGSE` | `01KSQRZ18QF1AY3TE4FKSGZJRZ` | trace | project:online | 382 -> 373 |
| 0.9512 | `01KRS1C3DYJP8YVZT3TF00SKMK` | `01KRS1E3XX8H48ZJ9678EBR849` | trace | project:octopus | 4880 -> 5005 |
| 0.9508 | `01KT9V2XFDW8SF5N0B5ZYHZR2J` | `01KT9SGNF0A9TS70NJK5K7WC3Z` | trace | project:online | 475 -> 516 |
| 0.9508 | `01KZ8S2JP2PMKQ4JSWQFATR1EE` | `01KWPN6CAQ3GEA6ZPDYF3EYPD4` | trace | project:ae | 1782 -> 1671 |
| 0.9504 | `01KTRECZGVHN4T07WBB0TTXDC5` | `01KTRJCB5FQ7ZT2WKD02TQ20B3` | trace | project:x | 200 -> 211 |
| 0.9502 | `01KTSKWP4F5CPNA5EB558XGVCW` | `01KSB3GDH7JCATXNGKP69TA7HG` | schema | project:octopus | 636 -> 917 |
| 0.9502 | `01KV8VK3HBTS21NG9APW522GKW` | `01KV8PQZQGJR91EBW2NXA7VBD7` | trace | project:x | 1709 -> 2436 |

## What would be SKIPPED, and why

```
## skip breakdown
  level_mismatch               223
  provenance_source_trace      847
  candidate_is_correction      3
  longer_than_bearer           37
  candidate_already_collapsed  237
```

1,347 of the 1,472 pairs are refused, and the two largest reasons are both the band
the root goal warned about, showing up above 0.95 rather than below it:

* **847 `provenance_source_trace`.** Every one of them is a trace/trace pair, and in
  827 of the 847 *both* traces are listed in `source_traces` of a live concept or
  schema. These are the near-identical traces a digest was built over; superseding
  either one rewrites the provenance of a live concept, so the cosine is irrelevant.
* **223 `level_mismatch`,** every one of them a schema next to a trace. This is the
  concept-over-its-sources similarity the goal measured at 0.82-0.95 — 223 pairs of
  it reach past 0.95, which is exactly why a threshold alone cannot be trusted to
  tell duplication from provenance.

### `provenance_source_trace` -- 847 pairs

the node that would be superseded is listed in source_traces of a live concept/schema: that band is provenance, never duplication

```
SKIP cos=1.0000  keep 01KRRPZ1FY4X74ZDHHGBMFXH6J  would-supersede 01KRT6BCRAREV4P15PF755S90T  (bearer-side collapse also blocked)
SKIP cos=1.0000  keep 01KRRPZ1FZ65GV0ANBMEY2QHZ1  would-supersede 01KRT6BCZVFS0PBSFJBFF6FX23  (bearer-side collapse also blocked)
SKIP cos=1.0000  keep 01KRRPZ1G6G3KS5MY1MRMWZXKF  would-supersede 01KRT6BDR5JGY39XNJ0R1F999D  (bearer-side collapse also blocked)
SKIP cos=1.0000  keep 01KRRPZ1H2XV718GEZ8W1M3H9W  would-supersede 01KRT6BFCXQBKMXYTYGZ7YF6M3  (bearer-side collapse also blocked)
SKIP cos=1.0000  keep 01KRRPZ1HJMTFPMBPWTWX4JE3T  would-supersede 01KRT6D89246BAC662HMH6SWFF  (bearer-side collapse also blocked)
SKIP cos=1.0000  keep 01KRRPZ1HKFAAA1N1C7YN821GT  would-supersede 01KRT6D8CVNE555PYXHBGGYTV2  (bearer-side collapse also blocked)
... 841 more
```

### `longer_than_bearer` -- 37 pairs

the candidate is materially longer than its bearer: same fact plus a new detail, and the detail has to reach the agent

```
SKIP cos=0.9959  keep 01KRVVF4H8V2NJWT2PDF2QWJ1X  would-supersede 01KSDT3SQS49QJGDRKTMHP207P  (7704 chars vs 6554 (+10% allowed))
SKIP cos=0.9941  keep 01KT7C5RXV3K98BS0H5WB4X5XR  would-supersede 01KT8735JAAMK3QZN0GXY6AXFJ  (70319 chars vs 36892 (+10% allowed))
SKIP cos=0.9924  keep 01KRXY78CFNNE6RP4T7STP82T1  would-supersede 01KT2968DY1VG786YGE3K3S7MW  (11384 chars vs 9299 (+10% allowed))
SKIP cos=0.9829  keep 01KS1NQ7GXFWAF5C8PWJS8EKG4  would-supersede 01KS1NQC1KS8SAHYBCTF0WXXRS  (353 chars vs 303 (+10% allowed))
SKIP cos=0.9799  keep 01KRVVE5XP69G159YJ94JTM020  would-supersede 01KRVVE5XQ9CJ7JPVPWFW9102F  (3650 chars vs 2081 (+10% allowed))
SKIP cos=0.9778  keep 01M0E0DZ5GD2NSTWN338N2FYPT  would-supersede 01M0FKQ19SXWTJG7ZMQWZQ4PGR  (1163 chars vs 974 (+10% allowed))
... 31 more
```

### `level_mismatch` -- 223 pairs

the two nodes are different levels (trace/concept/schema)

```
SKIP cos=0.9969  keep 01KTYWVB8XM9KRQWEHMFFM53BN  would-supersede 01KTZ5GK8NR2RMX6M58RYDPQY0  (trace vs schema)
SKIP cos=0.9918  keep 01KV8PQZQGJR91EBW2NXA7VBD7  would-supersede 01KVC1S1AC3TX8HJJ0SBQGVB03  (trace vs schema)
SKIP cos=0.9892  keep 01KTZ9NWFVHQF185TZ70A7R0Z7  would-supersede 01KV1F21NDJVYPW0Y6AZFHET0S  (trace vs schema)
SKIP cos=0.9885  keep 01KW08E88967RTW8V5XGZQZG4Z  would-supersede 01KW15GSTVDDHXHHSJBTPMXAKS  (trace vs schema)
SKIP cos=0.9877  keep 01KVCG14F663YBE1QN1JWQGPZF  would-supersede 01KVFRV25GRMDSVMW161NRXVRT  (trace vs schema)
SKIP cos=0.9872  keep 01KVZYYWRE0JR4B4PQRZHCGE75  would-supersede 01KVZZ99EF1R1639QZG6M01EG0  (trace vs schema)
... 217 more
```

### `candidate_is_correction` -- 3 pairs

the node that would be superseded is itself the source of a supersedes edge (it corrects something); collapsing it would bury the correction

```
SKIP cos=0.9844  keep 01KS81EPTT5AVNHRMA933ENKBC  would-supersede 01KSQHYDXAS360TN3M1K6V5JJH
SKIP cos=0.9736  keep 01KSQYEN0WDZJ7ECAYWE1H1WG8  would-supersede 01KS81EPTT5AVNHRMA933ENKBC
SKIP cos=0.9529  keep 01KSQYEN0WDZJ7ECAYWE1H1WG8  would-supersede 01KSQHYDXAS360TN3M1K6V5JJH
```

### `candidate_already_collapsed` -- 237 pairs

the candidate was already collapsed in this run

```
SKIP cos=1.0000  keep 01KY5HQ9QYAPMMTAV8T9074VRS  would-supersede 01KZ8RYAC3N836NFZ7DJBS03HM  (already superseded by 01KT7QN1JK2CAWM34QADEA3YCP in this run)
SKIP cos=1.0000  keep 01KY5HQ9QYAPMMTAV8T9074VRS  would-supersede 01KZEXTZJV576E1PH4J9ZQN8B4  (already superseded by 01KT7QN1JK2CAWM34QADEA3YCP in this run)
SKIP cos=1.0000  keep 01KY5HQ9QYAPMMTAV8T9074VRS  would-supersede 01KZZP9Q3YB1DZT0FHGZY17QSN  (already superseded by 01KT7QN1JK2CAWM34QADEA3YCP in this run)
SKIP cos=1.0000  keep 01KY5HQ9QYAPMMTAV8T9074VRS  would-supersede 01M0FAA12QJEG9181F01BX7M0P  (already superseded by 01KT7QN1JK2CAWM34QADEA3YCP in this run)
SKIP cos=1.0000  keep 01KY5HQ9QYAPMMTAV8T9074VRS  would-supersede 01M0FBSBYAN6BRMRWH67C2HSE4  (already superseded by 01KT7QN1JK2CAWM34QADEA3YCP in this run)
SKIP cos=1.0000  keep 01KY5HQ9VB4Q89YNA0GAVT6V89  would-supersede 01KZ8RYAFD04XJJFE50HHGSJHK  (already superseded by 01KT7QN1M6RD8NK4RZB5N5X9MR in this run)
... 231 more
```

## Invariants checked on this output

| check | result |
| --- | --- |
| nodes superseded that are in `source_traces` of a live concept | **0** |
| nodes superseded that are in `source_traces` of a live concept or schema | **0** |
| nodes both kept and superseded (a chain left dangling) | **0** |
| rows deleted / marked decayed by an apply | **0** -- the only write is an edge |
| second apply over the same database | 0 new edges, 125 pairs re-skipped |

The apply path was exercised on a throwaway duplicate of the copy
(`/tmp/lm-apply-test.sqlite3`): 125 edges written, node count 17,324 -> 17,324,
active 13,442 -> 13,442, connections 151,250 -> 151,375, chunk rows unchanged.

## numpy and no-numpy agree

The portability claim is that a host without numpy gets the same answer, slower.
Both backends were run over the full 13.4k-node copy:

| backend | scan wall time | collapses | skips |
| --- | --- | --- | --- |
| numpy 2.4.3 | 1.5 s | 125 | 1347 |
| pure Python (`LM_COLLAPSE_NUMPY=0`) | 84 s | 125 | 1347 |

The collapse lists are identical element by element, in the same order, and the skip
breakdowns match exactly. Cosines differ by at most 6.8e-07 (numpy accumulates the
dot product in float32, the pure path in float64), which is why pairs are ranked on
a cosine rounded to 5 decimals: without that, float noise inside a cluster of
byte-identical nodes decided which member became the bearer.

## Sensitivity: the two knobs that move the yield

| run | collapses | provenance skips | note |
| --- | --- | --- | --- |
| shipped default (`--scope-mode same`, `--provenance-levels concept,schema`) | 125 | 847 | the report above |
| `--scope-mode any` | 139 | 860 | median max-cosine 0.838 vs 0.832; collapses across scopes |
| `--provenance-levels concept` | 192 | 152 | schemas stop protecting their sources |

Same-scope pairing is the default because a supersedes edge is a corpus mutation:
a `project:x` node must not be superseded by a `project:lm` one that happens to
phrase the same sentence. Protecting schema sources as well as concept sources is
the default for the same reason -- a schema is a digest over a cluster exactly as a
concept is, so its similarity to its sources is provenance too. Both are one flag
away if the operator disagrees after reading the distribution.

## Reproducing this report

```bash
python3 - <<'EOF'
import sqlite3
con = sqlite3.connect("file:/home/sfx/.local/share/living-memory/global.sqlite3?mode=ro", uri=True)
con.execute("VACUUM INTO ?", ("/tmp/lm-neardup-copy.sqlite3",))
EOF
python3 scripts/lm_collapse_near_dups.py /tmp/lm-neardup-copy.sqlite3 \
    --threshold 0.95 --report-limit 0 --json /tmp/hygiene.json
```

