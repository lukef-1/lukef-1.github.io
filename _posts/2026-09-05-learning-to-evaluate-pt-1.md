---
title: Learning to Evaluate LLMs, part 1
---

*Building a basic evaluation testing LLM's ability to recall U.S. macroeconomic data.*

## Project goals

My goal for this project was to get more experience building LLM evaluations, and to document my learnings along the way. I subscribe to the view that building evals is [an increasingly important skill](https://freesystems.substack.com/p/an-army-of-citizens-building-evals) as LLMs become more engrained in our lives. What better way to learn that skill than by doing it.

This post walks through my experience getting a simple eval up-and-running using the [`Inspect`](https://inspect.aisi.org.uk) library and a custom dataset. I chose `Inspect` because it's a full-featured, open-source library used by many of the frontier labs. The results of my eval aren't revolutionary (or even close to it), which is okay! In the next post, I'll discuss how I'm refining and strengthening the eval.

#### Minor Notes
- [Repo with all code and logs!](https://github.com/lukef-1/fed_data_eval)
- A note on how I write my code: Since I'm studying computer science, I've hand-written 95%+ of the code in this project and only turn to LLMs for help when truly stuck. I don't want to outsource the learning and thinking.

## Evaluation design

**My eval tests LLMs' ability to accurately recall U.S. macroeconomic data from 2016 to 2026**. I chose this for a combination of personal interest and strong data availability. The [St. Louis Fed (FRED) API](https://fred.stlouisfed.org/docs/api/fred/) makes pulling economic data easy, meaning we can build a set of test questions with clear answers **and** build a tool for LLMs to pull data directly.

I was curious how the LLMs would perform on older and more recent data, so my questions spanned three year ranges: `2016 - 2020`, `2021 - 2025`, and `2026`. I expected good performance on older years (more likely in their training data) and little-to-no knowledge about `2026` unless given tool access.

The eval's data comes from 8 popular FRED data series, and each has a scoring tolerance to give credit for answers that are very slightly off. The tolerances weren't chosen scientifically, but I wanted them to be consistent across series with similar units. I then pulled three random observations per year range per series for **72** total samples (3 samples x 3 year ranges x 8 data series). Sampling made eval runs more manageable, but at the cost of less robust results. 

| Series ID | Series Name                                          | Units                                               | Tolerance |
| :-------- | :--------------------------------------------------- | :-------------------------------------------------- | :-------- |
| UNRATE    | Unemployment Rate                                    | Percent, Seasonally Adjusted                        | ± 0.1     |
| CIVPART   | Labor Force Participation Rate                       | Percent, Seasonally Adjusted                        | ± 0.1     |
| FEDFUNDS  | Effective Federal Funds Rate                         | Percent, Not Seasonally Adjusted                    | ± 0.05    |
| CPIAUCSL  | Consumer Price Index: All Urban Consumers, All Items | Index 1982-1984=100, Seasonally Adjusted            | ± 0.5     |
| CPILFESL  | CPI: All Items Less Food and Energy                  | Index 1982-1984=100, Seasonally Adjusted            | ± 0.5     |
| PAYEMS    | Total Nonfarm Payroll Employment                     | Thousands of Persons, Seasonally Adjusted           | ± 200     |
| HOUST     | New Privately-Owned Housing Units Started            | Thousands of Units, Seasonally Adjusted Annual Rate | ± 30      |
| INDPRO    | Industrial Production Index                          | Index 2017=100, Seasonally Adjusted                 | ± 0.5     |

I then asked each question to the LLMs in two scenarios:
1. `LLM Only`: LLM with no tools
2. `LLM + FRED API`: LLM with a tool to call the FRED API

(I originally had a third `LLM + Web Search` scenario, but it turns out web search uses a **ton** of tokens. I averaged 268,000 tokens per question in a small test run, so I removed that scenario to save some tokens.)

To keep things simple, I tested a recent flagship-ish model, Sonnet 5 (testing on a budget), and one of the strongest local models that runs on my laptop, Gemma 4: e4b. 


## Process: Building a basic evaluation

`Inspect` evals consist of a **Dataset**, a **Solver**, and a **Scorer**. Those map loosely to 1) what are you asking the model, 2) how is the model producing an answer, 3) and how are you grading that answer? Those categories provide a natural way to walk through my eval setup.

### Dataset: JSON file built using the FRED API
I started by writing a script that calls the FRED API, filters down to the sampled dates, and adds each result to a JSON file. That file gets loaded into each evaluation run. The `input` field has the prompt I constructed for the LLMs, the `target` field has the expected answer, and the remaining fields provide metadata to use in scoring and analysis. I added padding to each `tolerance` (e.g. 0.101 instead of 0.1) so answers on the boundary would be scored as Correct by float comparisons.

When building the `input` prompt, I leaned towards giving the LLM all the information it would need to answer the question. That included the FRED `series_id`, which is required for API calls, and the `units` expected in the response. This was one of the most important decisions in designing the eval, and including information liberally almost certainly made the task easier for the LLMs. More thoughts on that in the Takeaways section below.

After running the script and cross-checking the results on the St. Louis Fed website, we ended up with 72 entries like this:
```
{
    "input": "According to FRED series UNRATE (units: Percent, Seasonally Adjusted), what was the value of Unemployment Rate in the United States in July 2025?",
    "target": 4.3,
    "series_id": "UNRATE",
    "series_name": "Unemployment Rate",
    "units": "Percent, Seasonally Adjusted",
    "obs_date": "2025-07-01",
    "period_start": 2021,
    "period_end": 2025,
    "period_full": "2021-2025",
    "tolerance": 0.101
}
```


### Solver: System prompt + tools for each test scenario
Our solver varied by test scenario. The `LLM Only` scenario, as the name suggests, includes no external tools - the model is forced to generate an answer on the spot. `Inspect` has a built-in `generate()` function that triggers a final response from LLMs, so I used that function alone for this scenario.

`LLM + FRED API` includes a tool (i.e. a function I wrote and made available to the model) allowing the LLM to directly query the FRED API. That function has a docstring outlining the exact parameters needed to call the function, ideally making life easy for the LLM using it. 

```python
@tool
def call_fred_api():
    async def get_single_fred_value(series_id: str, date: str) -> str:
        """
        Returns the value for a specific FRED series in a specific month.

        Args:
            series_id: The FRED series ID to look up. Must be in abbreviated, all-caps format,
                e.g. GDPC1 to represent Real Gross Domestic Product. These IDs are provided
                directly by user prompts.
            date: The month to find data for in YYYY-MM-DD format. Days will always be "01".
                For example, "August 2026" is converted to "2026-08-01".

        Returns:
            A string listing the value for the series at the requested date.
        """
    ...
```

I also passed system prompts, which are appended to the question at run time, in the Solver step. Each system prompt defines the expected answer format (`ANSWER: <number>` or `ANSWER: UNKNOWN` if unsure), and the `LLM + FRED API` prompt tells the model about the available tool.


### Scorer: Regex parsing with tolerance checks
I started by using a pre-built scorer that checks if a target value shows up anywhere in the LLM response. While simple, that approach fell apart quickly. Number formatting varies across target values and LLM responses (e.g. `150937` vs. `150,937` and `4` vs `4.0`) and string matching treated the same numbers, but with different formatting, as incorrect.

Instead, I set up a custom scorer that uses regex to pull the number out of each response, converts it to a float, and scores it on whether the value falls within the series' tolerance. The regex parsing took the longest time of anything in the eval. I split the lowercased LLM response on the expected `answer:` phrase and took the first number that appeared after. I wrote a small test suite to make sure I covered the main edge cases.

The first regex parsing version was slightly broken and took a while to debug. I split on `answer` without the `:`, meaning that responses like "I don't have the answer for January 2016" were successfully split, then the regex scanned "for January 2016" and returned `2016.0`. Not ideal, but easily fixed once I identified the issue.

Also relevant to scoring, LLMs in this eval can reply with `UNKNOWN` if they don't know an answer. This is better than a wrong answer, but obviously still worse than a correct one. We'll only give credit for correct answers when grading accuracy, but `No Answer` will be split out from `Incorrect` when measuring the models' overall performance. More on this in the **Results** section.

### Running the Eval
These three components are combined into a single evaluation Task, like the `LLM + FRED API` scenario example below.

```python
@task
def fred_api_test_custom():
    return Task(
        dataset=json_dataset(
            "../questions.json",
            FieldSpec(
                input="input",
                target="target",
                id="question_id",
                metadata=["series_id", "series_name", "period_full", "tolerance"],
            ),
        ),
        solver=[system_message(TOOL_PROMPT), use_tools(call_fred_api()), generate()],
        scorer=within_margin(),
    )
```

I ran each sample through the eval **3 times** to make the results more robust since LLMs produce non-deterministic results. We'd need more than 3 runs for true robustness, but I'm trying to keep eval costs and time reasonable! 

After running everything, I used the built-in `Inspect` dashboard to view individual questions and responses, then wrote a quick script to aggregate the results on key testing dimensions. The dashboard shows an overview of all test samples and an in-depth view of each showing the model's process and final result.

## Results

Given the basic state of this eval, take the results with a major grain of salt. See the Takeaways section below for the improvements being considered.

> Another grain of salt: Standard errors below are optimistic given the question-level clustering. 

**Overall Accuracy, by Scenario** (Mean across all runs, n=216 per cell from 72 questions x 3 runs)

| Scenario | Accuracy | Standard Error |
| :---------------------------- | -------: | ----: |
| `LLM Only` (Sonnet 5) | 34.7% | ±3.2pp |
| `LLM + FRED API` (Sonnet 5) | **100%** | --|
| `LLM Only` (Gemma 4: e4b) | 1.9% | ±0.9pp |
| `LLM + FRED API` (Gemma 4: e4b) | 70.8% | ±3.1pp |

The `LLM + FRED API` scenario strongly outperformed `LLM Only`, with Sonnet 5 getting **100% of questions correct** when given access to the FRED API. Notably, Gemma 4: e4b went from a meager **1.9%** correct to **70.8%** once armed with the API. This result makes sense, since the API provides the exact answers to the tests, and asking an LLM to navigate a single tool call is simple. That said, Gemma 4: e4b struggled to use the tool at times, occasionally calling the wrong tool name and getting an error, or stating in its reasoning that it was going to call a tool and then doing nothing.


**Accuracy across Year Ranges, by Scenario** (Mean across all runs, n=72 per cell from 24 questions x 3 runs)

| Scenario | 2016 - 2020 | 2021 - 2025 | 2026 |
| :---------------------------- | ------------: | ------------: | -----------: |
| `LLM Only` (Sonnet 5) | 59.7% (±5.8pp) | 44.4% (±5.9pp) | 0.0% (--) |
| `LLM + FRED API` (Sonnet 5) | 100% (--) | 100% (--) | 100% (--) |
| `LLM Only` (Gemma 4: e4b) | 5.6% (±2.7pp) | 0.0% (--) | 0.0% (--) |
| `LLM + FRED API` (Gemma 4: e4b) | 68.1% (±5.5pp) | 77.8% (±4.9pp) | 66.7% (±5.6pp) |

Looking across year ranges, Sonnet 5 `LLM Only` performs best in `2016 - 2020`, slightly worse in `2021 - 2025`, and then gets nothing correct in `2026`. This matches the expectation that older data is more likely to be in the model's training corpus. Performance did not vary meaningfully across time in the `LLM + FRED API` scenario, with Sonnet 5 getting everything correct and Gemma 4: e4b's mistakes being caused by errors accessing the tool.

**Answer Type, by Scenario** (Mean across all runs, n=216 from 72 questions per cell x 3 runs)

| Scenario | Correct | Incorrect | No Answer |
| :---------------------------- | ------: | --------: | --------: |
| `LLM Only` (Sonnet 5) | 34.7% | 29.2% | **36.1%** |
| `LLM + FRED API` (Sonnet 5) | 100% | 0% | **0%** |
| `LLM Only` (Gemma 4: e4b) | 1.9% | 2.3% | **95.8%** |
| `LLM + FRED API` (Gemma 4: e4b) | 70.8% | 29.2% | **0%** |

Lastly, we'll look at the model's "refusal" rates, where replying `UNKNOWN` is treated as No Answer. Sonnet 5's `LLM Only` run produced No Answers in about a third of cases, largely driven by refusing to answer 100% of the 2026 questions. It still, however, produced incorrect numbers in about 30% of cases. Gemma 4: e4b refused to answer almost any question in its `LLM Only` run. 

Refusals are certainly better than factually incorrect answers, so the ideal split for `LLM Only` runs would be Correct + No Answer accounting for 100% of responses.


## Takeaways and next steps
This eval quantifies the huge impact the FRED API tool has on model performance. It's also interesting, but not surprising, that tool access significantly shrinks the performance gap between Sonnet 5 and Gemma 4: e4b. That said, an eval that can so easily be solved with isn't particularly useful for measuring model improvement or comparing performance across models.

My main focus for v2 of the eval is **making the task harder** to create more spread in results. I'm considering a few options, including no longer giving FRED `series_id`s in the prompt and shrinking the acceptable tolerance ranges. I'm going to do some research on other options too.

I also want to make the **eval conditions more robust**, and thankfully there are lots of easy improvements. We're only pulling three samples per time period, which may not be enough to separate true signal on performance from noise. I also didn't make any semantic modifications to prompts to see how models perform across different phrasing. Lastly, I'm only testing two models and not capturing exact model names and temperature settings. Some of my research time will be spent finding more options here.

**That's it for this post, more to come!** Thanks for reading.