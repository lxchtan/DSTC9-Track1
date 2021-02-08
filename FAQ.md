# DSTC9 Track 1 - Frequently Asked Questions

## Participation

Q: Do I need to complete any sign-up or registration process to participate this challenge track?

A: We strongly recommend you to complete the challenge registration at https://forms.gle/jdT79eBeySHVoa1QA by Sep 21. You will be asked to provide more details about your submission in the evaluation phase.

---

Q: Should I work on all three subtasks or can I participate in only one or two of them?

A: You will be asked to submit the full outputs across all three subtasks in the pipeline. If you’re interested in just one or two subtasks, we recommend to use the baseline model for the other tasks, so that you could make valid submissions.


## Data/Tasks

Q: Will the knowledge snippets for test set be released at later stage?

A: Yes, the additional knowledge snippets will be released along with the test dataset at the beginning of the evaluation phase.

---

Q: Will the domain API/DBs remain consistent for training and testing?

A: No, the backend contents will vary depending on the domain and locale of each conversation in the test set. We will release the API/DB entries on the new domain/locale along with the test dataset.

---

Q: Why do the data statistics and baseline performances differ from the task description paper?

A: Our task description paper reports the initial version of data and baseline models. An updated paper will be available after the evaluation phase.

---

Q: Can I assume that all the information in the knowledge snippets is outside the scope of API/DB?

A: Yes, that’s a valid assumption for all the data sets.

---
Q: How many knowledge snippets will be associated with each knowledge-seeking turn?

A: There will be only a single knowledge snippet relevant to each knowledge-seeking turn in all the data sets.
Every knowledge-seeking turn will have only a single relevant knowledge snippet.


## Evaluations

Q: Will I be allowed to manually annotate or preprocess the evaluation data/knowledge?

A: No, any manual task involved in processing the evaluation resources will be considered as a violation of the challenge policy. Please **DO NOT** touch the files in data_eval/ during the evaluation phase.

---

Q: Will I be allowed to manually examine my system outputs on the evaluation data to select the best model/configuration?

A: No, any manual examination on the evaluation data will be also considered as a violation of the challenge policy. Please freeze your models/systems only with the training/validation datasets and just submit the system outputs on the **unseen** evaluation data.

---

Q: Can I fine-tune or re-train my models on the unlabeled evaluation data in an unsupervised way?

A: No, you're not allowed to fine-tune or re-train your systems based on any evaluation resources. Please freeze your models/systems only with the training/validation datasets and just submit the system outputs on the **unseen** evaluation data.

---

Q: Can I fine-tune or re-train my models with the knowledge.json or db.json for the evaluation data?

A: No, you're not allowed to fine-tune or re-train your systems based on any evaluation resources. Please freeze your models/systems only with the training/validation datasets and just submit the system outputs on the **unseen** evaluation data.

---

Q: What is the final criterion of team ranking? Will we have rankings for each subtask or only one ranking as a final for all subtasks?

A: The official ranking will be based on **human evaluation** for the **end-to-end** performances. The human evaluation will be done only for selected systems according to automated evaluation scores.

---

Q: Will all the submissions be included in the final ranking by human evaluation?

A: No, the human evaluation will be done only for selected systems according to automated evaluation scores. The detailed selection criteria will be announced later with the automated evaluation results.

---

Q: Will the automated evaluation results be released also for the submissions which are not selected for the final ranking?

A: Yes, we will release the automated evaluation results for every valid submission with anonymized team IDs.

---

Q: How will the dependencies between the subtasks be considered in the end-to-end evaluation?

A: The scores.py calculates the end-to-end scores across all three subtasks in the pipeline, which differs from the component-level evaluations in the baseline training. The knowledge selection and the response generation scores are weighted by the knowledge seeking turn detection performances following the precision, recall, and f-measure concepts instead of just taking the average scores. The human evaluation scores will be calculated in the same way.

---
