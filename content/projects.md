+++
title = "Projects"
layout = "projects"
description = "Selected projects in federated analytics, agent workflows, and machine learning."

[[featured]]
name = "CohortLens"
label = "Federated analytics · Human-approved investigation"
description = "A statistical question travels to each site; the records stay there. Built on Flower, CohortLens combines local results and uses an LLM investigator to propose follow-up analyses when the sites disagree. Every follow-up needs human approval."
detail = "The demo uses public cancer cell-line data split across three emulated sites. Analysis contracts, query budgets, and a hash-chained ledger make each round inspectable."
stack = "Python / Flower / NumPy / LLM tools"
links = [{label = "Project explainer", url = "https://apzl.github.io/cohort-lens/"}, {label = "Source", url = "https://github.com/apzl/cohort-lens"}]

[[featured]]
name = "Lead Pipeline"
label = "Public-source research · Evidence-backed extraction"
description = "A pipeline that turns public sources into a ranked list of Berlin property managers. It discovers companies, collects websites and documents, and uses the Claude Agent SDK to extract facts with supporting source quotes."
detail = "Validation checks the extracted evidence before scoring and export. SQLite stores progress so interrupted runs can resume; a public dashboard presents redacted results."
stack = "Python / Claude Agent SDK / SQLite / Playwright"
links = [{label = "How it works", url = "https://apzl.github.io/case-study-dashboard/pipeline.html"}, {label = "Dashboard", url = "https://apzl.github.io/case-study-dashboard/"}, {label = "Source", url = "https://github.com/apzl/case-study"}]

[[earlier]]
name = "trackHN"
description = "Search Hacker News by a keyword's intended meaning, such as Go the language."
url = "https://github.com/apzl/trackHN"
demo = "https://trackhn.herokuapp.com/"

[[earlier]]
name = "QuoteBot"
description = "A motivational quote generator built with a fine-tuned GPT-2 model."
url = "https://github.com/apzl/QuoteBot"

[[earlier]]
name = "SANA"
description = "An audio classifier trained on the UrbanSound8K dataset."
url = "https://github.com/apzl/SANA"

[[earlier]]
name = "mask-or-not"
description = "An image classifier for checking whether a mask is worn properly."
url = "https://github.com/apzl/mask-or-not"
demo = "https://mask-ornot.herokuapp.com"

[[earlier]]
name = "what-to-watch-next"
description = "A movie recommender using collaborative filtering."
url = "https://github.com/apzl/Colab-filter"

[[earlier]]
name = "scoreX"
description = "A browser extension for live football scores and highlights."
url = "https://github.com/apzl/scoreX"
+++

Selected work in analytics, agent workflows, and machine learning.
