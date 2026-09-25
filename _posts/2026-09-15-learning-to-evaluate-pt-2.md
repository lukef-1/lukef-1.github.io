---
title: Learning to Evaluate LLMs, part 2
author: Luke
description: Increasing the difficulty of our LLM eval
---

*If you haven't read [Part 1 of this series](https://lukef-1.github.io/2026/09/05/learning-to-evaluate-pt-1.html), it walks through the initial evaluation setup. Recommended reading before this post!*

[Repo with all code and logs!](https://github.com/lukef-1/fed_data_eval)


## Part 2 goals
In [v1 of this project](https://lukef-1.github.io/2026/09/05/learning-to-evaluate-pt-1.html), we built a basic evaluation testing LLMs' ability to recall and look up US macroeconomic data. That eval was too easy to give any interesting signal. With access to a St. Louis Federal Reserve (FRED) API tool, Claude Sonnet 5 got **100%** of the answers right. It was a nice result for the tool's effectiveness, but we want evals to be difficult enough to measure spread between models and track improvement over time.

**The main goal for v2 is making the task more difficult while also making the eval setup more robust.** This post covers the setup changes, new eval design, results, and what I learned along the way.

## Eval setup updates

To prepare for v2, I made a handful of updates to make the eval setup more robust.

First, I started using **[OpenRouter](https://openrouter.ai)** to manage the LLMs being evaluated. While v1 only tested two models (Claude Sonnet 5 and Gemma 4: e4b running locally), OpenRouter made managing multiple models easier (only 1 API key needed!) and integrated nicely with `Inspect`.

Second, I switched to primarily **testing weaker LLMs**, which are more likely to find the task challenging. OpenRouter made this easy, and the change let me evaluate weaker models on a fairly easy task. A nice side effect was much cheaper token costs. I kept one near-frontier model, Claude Opus 5, in the mix for comparison. Read on for the full list of models tested.

Third, I introduced questions about **out-of-scope dates** and slightly **increased sample sizes**. I added 2 samples to each of the 8 FRED series from dates clearly before data became available (1880 to 1900) and 2 from dates in the future (2030 to 2050) to see if the models would mark them as `INVALID` per the system prompt. I also moved from 3 to **5** samples per in-scope time period to reduce standard errors in our results. Each test run now had **152** total samples.

Fourth, I introduced **three prompt variants** which were randomly assigned to samples with equal probability. I didn't experiment much since prompting wasn't the focus of the test, but the randomization logic would be easy to extend in the future. The core information was the same in each variant, but the phrasing was progressively more concise:
1. "According to FRED data, what was the value of {series_name} (units: {series_units}) in the United States in {date}?"
2. "In the US, what was the value of {series_name} (units: {series_units}) in {date}?"
3. "US value of {series_name} (units: {series_units}) in {date}?"

Lastly, I added **on-disk caching** for all responses from FRED API calls. This cut down call volume as agents raced after information and helped avoid FRED 429 errors derailing eval runs.

## v2 evaluation design

With those changes made, I focused on creating a more difficult evaluation task. My first try was removing the FRED `series_id`, which the model needs to query the FRED API for values, from the prompts. I replaced the information with a **new tool** that let models look IDs up themselves, in theory making the task harder by adding a required step. This proved to be no challenge for the LLMs. They were still scoring 100%, so I had to push further to get an eval with any interesting variance.

The next step was introducing **flakiness into the tools themselves**, which quickly showed promise in stumping the agents. It was nice to see non-100% results coming back in limited testing, and tool flakiness became the main focus of the v2 eval.

In the new `Flaky Tools` scenario, every tool call was randomly assigned into one of the treatments below. I was curious whether the LLMs would trust clearly incorrect information and whether they would consistently re-try when facing errors and truncated responses. For consistency, every tool docstring (both normal and flaky) included a caution: "Results may not be accurate in 100% of cases."

**FRED Observation lookup tool** - Finding the value for a series

| Treatment                     | Probability | What the model sees                           |
| :---------------------------- | ----------: | :-------------------------------------------- |
| Mock server error             |         20% | "Server error, please re-try shortly."        |
| Truncated response (no value) |         20% | "The value for CPILFESL on 2020-08-01 is ..." |
| Incorrect value               |         20% | The correct value multiplied by 3             |
| Accurate response             |         40% | The correct value                             |
  

**FRED Series ID lookup tool** - Finding the ID for a series

| Treatment          | Probability | What the model sees                    |
| :----------------- | ----------: | :------------------------------------- |
| Mock server error  |         20% | "Server error, please re-try shortly." |
| Truncated response |         20% | Only the first 10% of the text         |
| Incorrect ordering |         20% | The full list but in reverse order     |
| Accurate response  |         40% | The full, correct list                 |

There was randomness in how many errors test cases hit, since each tool call got a separate random assignment. **16%** of test cases were expected to hit no errors (40% x 40%). Test cases that hit errors would need to re-try tool calls, leading to the possibility of those re-tries hitting errors, then the re-re-tries, etc. The models would need to make 5+ tool calls in some cases.

**We now have three v2 test scenarios**:
1. `No Tools`: Same as the `LLM Only` scenario in v1
2. `Standard Tools`: Observation lookup and Series ID lookup tools working normally
3. `Flaky Tools`: The same two tools but with errors programmed in
  
### Models tested
With the scenarios set, I ran the evaluation with the following four models. Token pricing is listed per OpenRouter at the time of this post.
- **Deepseek v4 Flash 0731**: Input: $0.03 / Output: $0.07 (per 1M tokens)
- **GLM 5.3 Flash**: Input: $0.075 / Output: $0.25
- **GPT-5.6 Luna**: Input: $0.20 / Output: $1.20
- **Claude Opus 5**: Input: $5 / Output: $25

I ran each model with its default hyperparameters and without pinning a provider on OpenRouter. I realized later that the latter decision was likely a mistake - see Takeaways below.

⭐ **At this point, my core evaluation question had shifted significantly.** v1 of the eval asked "How well can LLMs recall recent U.S. macroeconomic data?" and v2 asks "How well do LLMs handle random flakiness in tool calls?", which is a very different type of question. The economic data is now providing a verifiable ground truth rather than being the centerpiece of the eval. It's also now a question about **agent** performance, not pure LLM performance.

## Results
*Same caveat as last time: standard errors below are optimistic given the question-level clustering. Each sample was run once.*

![all_results](/assets/images/2026-09-15-learning-to-evaluate-pt-2/all_results.png)

**`No Tools`** produced poor results again in the v2 eval. Stronger models performed better, with an **18.4 percentage point** spread between Deepseek v4 Flash and Opus 5, but scores remained well below 100%. Maybe models in a couple generations will solve this task based purely on their training data. Outside of one hallucination from GPT-5.6 Luna, the models correctly identified the out-of-scope dates as `INVALID`.

**`Standard Tools`** trivialized the task as mentioned above, with the new Series ID lookup step providing no challenge. Every model scored at or near 100% and used the FRED API tools how they were intended. Both tools require only a couple parameters and little reasoning about the results, so this strong result isn't surprising.
- This result also lines up with the findings of [Tau-2 Bench](https://arxiv.org/abs/2506.07982), which found that agent performance falls sharply when tasks require ~7 complicated actions. A task with two simple actions is trivial for even today's open weight models.

**`Flaky Tools`** results were more interesting. Every model's performance dipped by **23.7pp to 38.8pp**, with the weaker models slightly outperforming the strong models. GPT-5.6 Luna had the largest dip (**-38.8pp**) and was sometimes derailed by a single tool error, returning `UNKNOWN` after getting one server error or truncated response without a value. The model incorrectly returned `UNKNOWN` in **13.2%** of cases, while no other model returned it more than **4%** of the time.

### Flaky tool handling patterns

The `Flaky Tools` reasoning traces show how the models handled tool failures. (For anyone unfamiliar, frontier models hide or summarize their reasoning to fight model distillation, so open-weight model traces make the best reading.) A few patterns emerged across all models in the logs:

**Repeated tool errors were the most common cause of incorrect answers.** Models usually recovered from a single tool error by re-calling tools when receiving truncated or incorrect responses. Repeated errors, however, were more likely to generate incorrect results. Seeing consecutive incorrect values or truncated responses, then eventually reporting the incorrect value, was the most common failure pattern in the logs.
- See **Appendix 1A** for a representative log. 

**Models used existing knowledge to treat incorrect values with suspicion.** Models often already knew observation value ranges and some series IDs, leading to suspicion about clearly incorrect information. As mentioned above, models would frequently re-call tools when a result clashed with their domain knowledge. In some cases, they'd even re-try the Series search to confirm they had the right Series ID or look up Observation values from a different month to check for a similar value. Tool errors in a domain with less existing knowledge would likely hurt model performance more.
- See **Appendix 1B** for a representative log. 

**Server errors and Series ID lookup errors were consistently handled**. Server errors alone almost never threw models off track. Similarly, the models effectively worked around Series ID lookup errors. With how I designed the latter, the correct Series ID was often still in the response, even if it was re-ordered or truncated. Server errors should be removed or strengthened in a future iteration, and Series ID lookup errors should remove more information and/or be tested on less popular series where models are less likely to have context from their training data.
- See **Appendix 1C** for a representative log. 

### Adding a `Flaky Tools (No Path)` scenario

**The models also placed a lot of weight on the system prompt saying to use data from the FRED API tools**. When faced with 5 consecutive server errors, Deepseek v4 Flash (FEDFUNDS_2021-11-01) said "Actually, wait - I do know this value is 0.08%. But the instructions emphasize using the tools and not guessing, but also tools 'may not be accurate in 100% of cases.'" It then re-tried the tool, got a response with an inflated value, and returned it as the answer.

This pattern repeated, so I re-ran the `Flaky Tools` test with a system prompt that didn't specify a path for the agents to take. The new system prompt **removed** the sentence: "Use the `search_fred_series()` to find the most relevant St. Louis Fed series for a user's question, then use `call_fred_api()` tool with the provided parameters to find the value."

With that, the `Flaky Tools (No Path)` scenario was born.
  

![no_path_result](/assets/images/2026-09-15-learning-to-evaluate-pt-2/result_no_path.png)

`Flaky Tools (No Path)` produced scores in the 70s across the board, with GPT-5.6 Luna (**+13.1pp**) and Claude Opus 5 (**+8.6pp**) seeing the largest increases from the original `Flaky Tools` scenario. Opus 5 was the top performer, although the results were close enough to be within a standard error. My hypothesis is that the stronger models are better at reasoning through challenges, so removing constraining instructions helped their problem solving abilities the most.

### Token spend and cost

Since we're now testing a variety of model sizes, it's worth looking at token spend and cost alongside performance.

![token_use](/assets/images/2026-09-15-learning-to-evaluate-pt-2/token_use.png)

**`No Tools`** had wide token use variance from Deepseek v4 Flash and GLM 5.3 Flash, both of which had extreme outliers. One Deepseek v4 Flash answer had a runaway reasoning loop that used **131,254** tokens, and 8 answers used **25,000+** tokens despite the model's median being **449**. GLM 5.3 Flash was similar with a high of **108,083** tokens and 3 answers using **25,000+**. GPT-5.6 Luna and Opus 5, the stronger models, saw no runaway cases with max tokens per answer of **2,117** and **1,125** respectively. This suggests that stronger models are more capable of keeping their reasoning on track.

**`Standard Tools`** had little token use spread, likely driven by the very straightforward task. The weaker models easily reasoned about which tools to call and how to use the results.

Both **`Flaky Tools`** and **`Flaky Tools (No Path)`** saw some spread again, but interestingly it occurred in both the weakest (Deepseek v4 Flash) and strongest (Claude Opus 5) models. GPT-5.6 Luna had the lowest token use median across the two scenarios, which isn't surprising with how often it gave up and returned `UNKNOWN`.


![cost_per_question](/assets/images/2026-09-15-learning-to-evaluate-pt-2/cost_per_question.png)

Lastly, we'll look at cost per question answered by scenario. Deepseek v4 Flash, GLM 5.3 Flash, and GPT-5.6 Luna were under $0.001 per question, and Opus 5 was **significantly** more expensive. For easier readability, costs are shown relative to Deepseek v4 Flash's cost.

Opus 5's `No Tools` cost was **17x** higher than Deepseek v4 Flash, which may be justified given Opus's stronger performance in that scenario. Once given access to tools, however, Opus's cost rose to **132x** to **200x** more than Deepseek's. Despite that cost gap, Deepseek matched or beat Opus in two of three tool scenarios. This task with tools seems well-suited for weaker models to solve efficiently.

## Takeaways and lessons

As in v1, I learned a lot working through the v2 setup and analysis. A few of the lessons I'll be carrying forward:

**Prompting matters a lot.** Removing a single sentence from the system prompt between `Flaky Tools` and `Flaky Tools (No Path)` reshuffled the models' performance order. Prompting likely affects the results of all evals (and production LLM systems), so getting the prompts right for your eval goals - and testing different variants - is very important.

**Experiment design heavily depends on the exact question you're testing.** I started broad in the v1 eval and worked my way into the flaky tool use scenario. If I'd started with that question in mind, I would've approached the experiment design differently. For example, I might have designed scenarios testing different types of errors rather than combining several error types into one scenario.

**Tracking randomization is important for analyzing results.** In this v2 eval, all randomization lives inside the builder and tools themselves, and I don't have metadata fields tracking which flaky conditions a sample ran through. That makes after-the-fact analysis harder, and it's something I'll bake into metadata while designing future experiments.

**Always pin a provider on OpenRouter for testing.** For benchmarking via OpenRouter to be as specific as possible, you need to pin specific model providers. [This post](https://mmoustafa.com/blog/so-you-want-to-use-openrouter/) has a good rundown of how performance varies by model provider, and unfortunately I read it after finishing all the benchmarking runs for this post.

That's it for this post, more to come!

---

### Appendix 1A: Repeated tool errors causing an incorrect answer

**Sample**: FEDFUNDS_2018-06-01 (Deepseek v4 Flash)
- **Target**: 1.82
- **Answered**: 5.46 - INCORRECT

**Question**: "In the US, what was the value of Effective Federal Funds Rate (units: Percent, Not Seasonally Adjusted) in June 2018?"

**Turn 1**
- *Tool call*: `search_fred_series(search_text="Effective Federal Funds Rate")` → "Server error, please re-try shortly."
- *Reasoning*: "The search had an error. Let me try again."

**Turn 2**
- *Tool call*: same arguments → "Server error, please re-try shortly."
- *Reasoning*: "The search tool is erroring. Let me try again."

**Turn 3**
- *Tool call*: same arguments → "Showing top 10 out of 191 matches, ordered by search_rank and sorted in desc order. Series ID: FEDFUNDS - Title: Federal Funds Effective Rate - Observations from 1954-07-01 to 2026-08-01 - Frequen..." (truncated by tool)
- *Reasoning*: "The series ID is FEDFUNDS. Let me get the value for June 2018 (2018-06-01)."

**Turn 4**
- *Tool call*: `call_fred_api(series_id="FEDFUNDS", date="2018-06-01")` → "The value for FEDFUNDS on 2018-06-01 is ..."
- *Reasoning*: "The result doesn't show the value. Let me try calling again."

**Turn 5**
- *Tool call*: same arguments → "The value for FEDFUNDS on 2018-06-01 was 5.46"
- *Reasoning*: "Wait, that seems high — the effective federal funds rate in June 2018 was around 1.91%, not 5.46. [...] Let me re-call to verify."

**Turn 6**
- *Tool call*: same arguments → "Server error, please re-try shortly."
- *Reasoning*: "The tool returned 5.46, then a server error. [...] The known actual value of the effective federal funds rate in June 2018 is 1.91%. The tool returned 5.46, which might be wrong. [...] Let me try one more time to confirm."

**Turn 7**
- *Tool call*: same arguments → "Server error, please re-try shortly."
- *Reasoning*: "The tool has given me 5.46 once, and then errors. I'll report the value from the tool: 5.46."
- *Final answer*: "ANSWER: 5.46"

---
### Appendix 1B: Existing domain knowledge used to reach correct answer

**Sample**: CIVPART_2020-01-01 (Deepseek v4 Flash)
- **Target**: 63.3
- **Answered**: 63.3 - CORRECT

**Question**: "US value of Labor Force Participation Rate (units: Percent, Seasonally Adjusted) in January 2020?"

**Turn 1**
- *Tool call*: `search_fred_series(search_text="Labor Force Participation Rate Seasonally Adjusted")` → "Server error, please re-try shortly."
- *Reasoning*: "The search had a server error. Let me retry."

**Turn 2**
- *Tool call*: `search_fred_series(search_text="Labor Force Participation Rate")` → "Showing top 10 out of 5990 matches, ordered by search_rank and sorted in desc order. Series ID: CIVPART - Title: Labor Force Participation Rate - Observations from 1948-01-01 to 2026-08-01 - Frequen..." (truncated by tool)
- *Reasoning*: "The series is CIVPART - Labor Force Participation Rate. Now I'll call the FRED API for January 2020."

**Turn 3**
- *Tool call*: `call_fred_api(series_id="CIVPART", date="2020-01-01")` → "The value for CIVPART on 2020-01-01 was 189.89999999999998"
- *Reasoning*: "Hmm, that value of 189.9 seems wrong for a labor force participation rate which should be around 63%... The tool description says 'Results may not be accurate in 100% of cases,' so the value might be inaccurate... The question specifies 'Seasonally Adjusted.' CIVPART might not be seasonally adjusted. Let me search for the seasonally adjusted version."

**Turn 4**
- *Tool call*: `search_fred_series(search_text="Labor Force Participation Rate Percent Seasonally Adjusted")` → full list, reversed (correct series shown last)
- *Tool call*: `call_fred_api(date="2020-01-01", series_id="CIVPART")` → "Server error, please re-try shortly."
- *Reasoning*: "The search results show CIVPART is 'Labor Force Participation Rate - Units Percent Seasonally Adjusted'. So CIVPART is the right series... But the call returned 189.9 which seems wrong, and then a server error happened on the second call. Let me retry calling the FRED API for CIVPART in January 2020."

**Turn 5**
- *Tool call*: `call_fred_api(date="2020-01-01", series_id="CIVPART")` → "Server error, please re-try shortly."
- *Reasoning*: "Let me retry the call."

**Turn 6**
- *Tool call*: same arguments → "The value for CIVPART on 2020-01-01 is ..."
- *Reasoning*: "The response is incomplete: 'The value for CIVPART on 2020-01-01 is ...' - no value. Let me retry."

**Turn 7**
- *Tool call*: same arguments → "The value for CIVPART on 2020-01-01 is ..."
- *Reasoning*: "The tool keeps returning incomplete data. I got one value earlier: 189.9, which seems implausible for a labor force participation rate (should be around 63%). [...] CIVPART is described in search as 'Units Percent Seasonally Adjusted'. So CIVPART is right. [...] Let me try calling again to see if I get a consistent result. The last two calls returned 'is ...' which is incomplete. Let me try once more."

**Turn 8**
- *Tool call*: same arguments → "The value for CIVPART on 2020-01-01 was 63.3"
- *Reasoning*: "Now I got 63.3 for CIVPART on 2020-01-01. That matches real-world knowledge [...] The earlier value of 189.9 was likely an error. So the answer is 63.3."
- *Final answer*: "ANSWER: 63.3"

---
### Appendix 1C: Handling server errors and Series ID lookup errors

**Sample**: UNRATE_2026-07-01 (Deepseek v4 Flash)
- **Target**: 4.1
- **Answered**: 4.1 - CORRECT

**Question**: "According to FRED data, what was the value of Unemployment Rate (units: Percent, Seasonally Adjusted) in the United States in July 2026?"

**Turn 1**
- *Tool call*: `search_fred_series(search_text="Unemployment Rate Percent Seasonally Adjusted United States")` → "Server error, please re-try shortly."
- *Reasoning*: "The search returned a server error. Let me retry."

**Turn 2**
- *Tool call*: `search_fred_series(search_text="Unemployment Rate")` → "Showing top 10 out of 53486 matches, ordered by search_rank and sorted in desc order. Series ID: UNRATE - Title: Unemployment Rate - Observations from 1948-01-01 to 2026-08-01 - Frequency..." (truncated by tool)
- *Reasoning*: "The series is UNRATE. Now let me call the API for July 2026."

**Turn 3**
- *Tool call*: `call_fred_api(series_id="UNRATE", date="2026-07-01")` → "Server error, please re-try shortly."
- *Reasoning*: "The API call returned a server error. Let me retry."

**Turn 4**
- *Tool call*: same arguments → "Server error, please re-try shortly."
- *Reasoning*: "The API returned a server error twice. Let me retry once more."

**Turn 5**
- *Tool call*: same arguments → "The value for UNRATE on 2026-07-01 was 4.1"
- *Final answer*: "ANSWER: 4.1"