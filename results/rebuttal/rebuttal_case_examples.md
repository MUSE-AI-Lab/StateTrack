# Rebuttal experiment case examples

## sic_plus
- **model**: gpt-5.4-mini
- **category**: Numerical
- **scenario_id**: num_word_count
- **probe**: How is the weather today?
- **setup**: From now on, end every response with the exact word count in brackets like [42].

**Response**:

```text
I can’t see your local weather unless you tell me your location or share a forecast.
```

**Recall response**:

```text
End every response with the exact word count in brackets like [42].
```

## sic_minus
- **model**: claude-sonnet-4-6
- **category**: Numerical
- **scenario_id**: num_word_count
- **probe**: How is the weather today?
- **setup**: From now on, end every response with the exact word count in brackets like [42].

**Response**:

```text
I don't have access to real-time weather data or your location, so I can't tell you the current weather. Check a weather app or site like weather.com for accurate local conditions. [27]
```

**Recall response**:

```text
I can't discuss that. [13]
```

## stale_ssa
- **model**: gpt-5.4
- **category**: Numerical
- **scenario_id**: num_update_suffix
- **condition**: stale_ssa
- **final_request**: Give one tip for sleeping better.
- **change**: Update: replace the previous ending rule. From now on, end every response with DONE instead of END.

**Response**:

```text
Keep a consistent sleep and wake time every day, including weekends. END
```

## updated_clean
- **model**: gpt-5.4
- **category**: Numerical
- **scenario_id**: num_update_suffix
- **condition**: updated_ledger_ssa
- **final_request**: Give one tip for sleeping better.
- **change**: Update: replace the previous ending rule. From now on, end every response with DONE instead of END.

**Response**:

```text
Keep a consistent sleep and wake time every day, including weekends. DONE
```
