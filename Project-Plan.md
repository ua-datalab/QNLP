# References and links:
* Link to github started by Robert Henderson: [here](https://www.google.com/url?q=https://github.com/bkeej/usp_qnlp&sa=D&source=editors&ust=1717607867014854&usg=AOvVaw3ji0W3TH7OhJaizgZHp14m)
	* QNLP dataset: [https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data](https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data)
 * QNLP code repo: [https://github.com/ua-datalab/QNLP/blob/main](https://github.com/ua-datalab/QNLP/blob/main)
* Link to white paper: [https://www.overleaf.com/4483532232tcfnfdrrcbdc#12a1b4](https://www.google.com/url?q=https://www.overleaf.com/4483532232tcfnfdrrcbdc%2312a1b4&sa=D&source=editors&ust=1717607867015283&usg=AOvVaw0VwgWn_tu2jNMuTmaj2PDL)
* All data (e.g. spanish only files) is stored in a [gdrive folder here](https://www.google.com/url?q=https://drive.google.com/drive/folders/1m4nFZwsUcZ2DQzN3nYaK0_oKJXGhV575?usp%3Ddrive_link&sa=D&source=editors&ust=1717607867015673&usg=AOvVaw32Cbwsxm70wOGxbbRLFbb0)
	- Uspantekan data: [https://drive.google.com/drive/folders/1CtMhTf-v0nSUSaTJVelILkDMrLfF1U5Y?usp=share\_link](https://www.google.com/url?q=https://drive.google.com/drive/folders/1CtMhTf-v0nSUSaTJVelILkDMrLfF1U5Y?usp%3Dshare_link&sa=D&source=editors&ust=1717607867016039&usg=AOvVaw3cDmd4Rclx66QuxHrZGi-b)
* Jira Link: [https://cyverse.atlassian.net/jira/software/projects/QNLP/boards/27](https://www.google.com/url?q=https://cyverse.atlassian.net/jira/software/projects/QNLP/boards/27&sa=D&source=editors&ust=1717607867016357&usg=AOvVaw2fccm9pIgF5Yw5sAb26eH0)         
* 1998 Lambeks paper on math===language: [here](https://www.google.com/url?q=https://drive.google.com/file/d/1BWhs5zOoA2n7y8aUnKoamfift0t9Xdhu/view?usp%3Dsharing&sa=D&source=editors&ust=1717607867016692&usg=AOvVaw3z0FGavJsHiiA0aRD5yFLn)
* 2020 bob coecke [QNLP](https://www.google.com/url?q=https://drive.google.com/file/d/15hXA_ecFN31JJdt9E8POdUFT1mlcwssv/view?usp%3Dsharing&sa=D&source=editors&ust=1717607867017007&usg=AOvVaw2jRw8msgoQEVE_z5vZxQCa)
* Type grammar revisited: [https://link.springer.com/chapter/10.1007/3-540-48975-4\_1](https://www.google.com/url?q=https://link.springer.com/chapter/10.1007/3-540-48975-4_1&sa=D&source=editors&ust=1717607867017296&usg=AOvVaw0T99YvALpGGqp50dAnYxz9)
* Khatri et al. thesis: [https://github.com/ua-datalab/QNLP/blob/main/OOV_MRPC_paraphrase_task.ipynb](https://github.com/ua-datalab/QNLP/blob/main/OOV_MRPC_paraphrase_task.ipynb)
* Colab notebooks:
	- Data pipeline: [https://colab.research.google.com/drive/1YwdVkFZRt30QPuUwQ-y9W1vSnYlkS656?usp=sharing](https://www.google.com/url?q=https://colab.research.google.com/drive/1YwdVkFZRt30QPuUwQ-y9W1vSnYlkS656?usp%3Dsharing&sa=D&source=editors&ust=1717607867017671&usg=AOvVaw3sePnYQ_2mwLcqo1YYvu9Y)
	- Lambeq for Spanish, run with Spider parser: [https://drive.google.com/file/d/1wTo8rAObpuLu65DyFo1D0gE5kKjUtzBf/view?usp=sharing](https://www.google.com/url?q=https://drive.google.com/file/d/1wTo8rAObpuLu65DyFo1D0gE5kKjUtzBf/view?usp%3Dsharing&sa=D&source=editors&ust=1717607867017990&usg=AOvVaw1oMypNSQtjg_K-olMfxRnv)

# Working Group Best Practices
- Save all data in the private repository, to prevent leaks: [https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data](https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data)
- Save code to public repository, so it can be opened on Cyverse and run on colab: [https://github.com/ua-datalab/QNLP/blob/main/](https://github.com/ua-datalab/QNLP/blob/main/OOV_MRPC_paraphrase_task.ipynb)
- Cyverse has resource allocations- so all big training done there. Example: 

# Project plan 
Goal: we want to show Robert a proof of concept that QNLP can work with uspantekan- limited resources- and still give good accuracy
1. Can qnlp + uspantekan- straight out of the box give a good classification accuracy- if yes:
	1. Path 1 bottom up:  
		1. Pick one thread. Eg. spider
			1. Trying on spanish
			2. Find embedding spanish
				1. Split by space- what is accuracy
				2. Try splitting - Spanish tokenizer- did accuracy improve
			3. Align with embedding uspantekan
			4. Run classification task again on uspantekan
	2. Path 2
		1. Qn- why don‘t we straight Run classification task again on uspantekan -with spider
			1. No parser which can break tokens faithfully
			2. No embeddings directly for uspantekan
			3. How much is bare bones accuracy?
			4. With tuning how much can you get it upto?
			5. If they both fail,
			6. Then yes, we can think of bringing in spanish embeddings.
			7. Todo
				1. Train dev of uspantekan
				2. Modifying
2. update: june 5th 2024
	1. Path 2: Ran experiment. [here](https://docs.google.com/spreadsheets/d/1NBINiUsAdrqoO50y_CX_BGGgXcP9Zt6i5nYKvuB70Tg/edit?usp=sharing) are the results of trying Uspantekan with spider parser, bobcat parser, and using pytorch trainer. Rather after tuning max accuracy on dev was 72%...which was given by cups and stairs model -so we have decided to move on.
	2. Options to explore next
		1. path 1: go back to  spanish- and load with embeddings.
		2. path 2: try with discocat/bobcar parser + uspantekan.. last time we tried, got errors...
3. update: june 25th 2024:
	- still working on QNLP +uspantekan + embedddings. reevaluated goal and pivoted because of the question: what baseline
	- are we trying to beat. Decided we will do the baseline first on LLMs
      
# General correspondence:
* why did we decide to go with spanish first and not Uspanthekan?
	- Lambeq pipeline seems to have a language model requirements and needs embeddings. We have some for Spanish, none for Uspantekan
	- We have direct Uspanteqan-Spanish translations, but not English-Uspanteqan. Which means that if things fail, we have no way to examine what happened if we used an English model.



# Meeting Notes

## July 25th 2024
- Looked at the Khatri et al. code for Spanish, and worked on code fix: https://github.com/ua-datalab/QNLP/tree/megh_dev
	-  Most of the word are OOV, so they can't be put into grammatical categories
	-   
- General Discussion:
	-  How does Quantum trainer compare to NN?
 		- Feed forward- we look at the loss between the original and predicted value, does back propagation, until the right combination of weights provides us useable prediction
  		- Instead of neurons, we use quantum circuits
  		- Thus, the trainer for QC is the same as NN- both have similar black boxes
- The code should look the same for deep and quantum trainer  

## June 25th 2024
- Pivot to actually determining what the LLM baseline classification accuracy is for our dataset, os that we know what the quantum approach needs to beat.
- ToDo Megh:
	- Move Mithun's single python file to our repo
 	- Edit to load Uspantekan data 
	- Run the LLM code, an untrained DistilBERT and RoBERTa using Mithun's codebase
	- Report bugs and/or result.
- Mithun tries to load embeddings + quantum
  
## June 21st 2024
- Updates to khatri et al code:
	- Current work on Spanish data, using khatri et. al.: [https://github.com/ua-datalab/QNLP/tree/megh_dev](https://github.com/ua-datalab/QNLP/tree/megh_dev)
 	- Mithun shared his updated code for khatri et. al., that works on Uspantekan:  https://github.com/ua-datalab/QNLP/tree/mithun_dev
  - Overhauled code to fit our classification task that has only one feature vector, as opposed to two. `lambeq` libraries and modules needed to be replaced due to depreciation.

## June 7th 2024

* How to access shared analyses on Cyverse- Mithus shares his setup so we have access to GPUs
* Go to [https://de.cyverse.org/analyses/](https://de.cyverse.org/analyses/) and switch the dropdown menu to "shared with me"
* Choose the relevant analysis.
* Setup and tech issues
	* Get permission to run analysis with a GPU on cyverse. Talk to Michele about permissions
 	* Set CPU to 8GB 

## June 5th 2024
* [results location]([url](https://docs.google.com/spreadsheets/d/1NBINiUsAdrqoO50y_CX_BGGgXcP9Zt6i5nYKvuB70Tg/edit?usp=sharing))
1. Can qnlp + uspantekan- straight out of the box give a good classification accuracy?
	* update: NO
	* Rather after tuning max accuracy on dev was 72%...which was given by cups and stairs model
1. Options
	1. path 1:
		1. go back to  spanish- and load with embeddings.
	2. path 2: try with discocat/bobcar parser + uspantekan.. last time we tried, got errors.
* Updated Jira with new tasks and updates
* Megh's updates:
	* Set up repo for code, as well as a folder with dataset in rhenderson's repo:
 	* QNLP dtaset: https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data
 	 * QNLP code repo: https://github.com/ua-datalab/QNLP/blob/main/Project-Plan.md	 
* Todo for Megh
	* Run the code for Spanish- we will need embeddings
	* Use fasttex- Used by GPT for building ngrams and aligned work vectors. 
	* Thesis code from Mithun’s contact (Khatri et al.)
		* Run this code directly and see what happens
* Examined issues received outlook calendar nvotes

## May 30, 2024 :
* Jira setup for project
* Classical model pipeline with Spanish up and running
	- Major update- removed all sentences with >32 tokens, and the Spanish model was able to run with lampeq
	- Training accuracy is good, but loss is too high
	- Needs finetuning
* What is the language model in th classical case doing?
	- Spanish SpaCy model is using the tokenizer to be able to use word-level knowledge.
	- Then, bobcat uses the sentence2diagram method on the result.

* New proposal- to run Spider directly on the uspantekan sentences
* ToDo- move notes to Github, create notes.md and keep all data on Github
* Jira for project setup-
	- Todo- assess the space and make sure Megh knows her tasks.
	- Check out QNLP52
* Paper writing
	* Pivot from a whitepaper to a one-pager
		* Motivation, contribution thought

## May 7th, 2024:
* Classical case:
	- Filename non-ASCII issue resolved with rename   and train, test, dev splits saved
	- Classical case-  [ran into error while running sentence2diagram](https://www.google.com/url?q=https://colab.research.google.com/drive/12kNxLNX162hGznIYenBSqLJbflmFaE1y?usp%3Dsharing&sa=D&source=editors&ust=1717607867019672&usg=AOvVaw1SAvjipfXEAOkwHKcnRCgQ)
	- Check classical case with Spanish, assign spanish SpaCY language model to pipeline if needed
	- ToDo for Mithun
	- Fix classical trainer, section “Training the model”, cell 67
* Whitepaper updates: Mithun shared his current working document

## April 19th, 2024:
* Issue with non-ASCII characters in filename persists- can’t read from file
* Pipeline for reading sentences ready- need to work on reading files in with non-ASCII filenames
* Todo:
	- Mithun- set up an overleaf for a paper draft
	- Mithun- find out how to run QUANTUM CASE on Cyverse
* Data pipeline todo \[Megh\]:
	- randomize sentences, and create test, val, train splits
	- Convert the sentences to a dataset in the format: Label\[Tab\]Text\[Tab\].
		+  0 for bailar, 1 for educacion
* ToDo Mithun: create list of file names, and convert them to ASCII

## April 11th, 2024:

* Updates
	- Classical case set up on Cyverse
	- Data cleaning pipeline setup
* ToDo: upload code to Colab \[DONE\]
* Todo:
	- Mithun- set up an overleaf for a paper draft
	- Mithun- find out how to run QUANTUM CASE on Cyverse
	- Megh- find a way to run data pipeline on Cyverse \[DONE\]
	- Megh- complete data cleanup pipeline \[DONE\]
* Proof of concept:
	- Data- done
	- Data cleanup pipeline: done
	- Code- quantum case in process
* Data pipeline todo: \[DONE\]
	- randomize sentences, and create test, val, train splits
	- Convert the sentences to a dataset in the format: Label\[Tab\]Text\[Tab\].
		+  0 for bailar, 1 for educacion
	- ToDo Mithun: create list of file names, and convert them to ASCII
* QNLP pipeline setup
	- ToDo Megh: Run classical case on Cyverse with Spanish data

## April 2nd, 2024:

1. We went through learning
	1. Bobcat parser
	2. Convert text to diagrams
	3. Convert diagrams to circuits
	4. Rewriter+ reduce cups
	5. Go through full fledged training example of classification of IT or ood
2. Todo \[Megh\] for next week:
	1. Set up and Try out lambeq classification [task](https://www.google.com/url?q=https://cqcl.github.io/lambeq/tutorials/trainer-classical.html&sa=D&source=editors&ust=1717607867023254&usg=AOvVaw3Se6n9TAlMX5oyEHCO2C1Z)  on Cyverse
	2. With spanish text from the Robert henderson’s data
		1. Pick 2 classes (i.e file names in the data directory)
			1. dancing/bailes
			2. [Education](https://www.google.com/url?q=https://github.com/bkeej/usp_qnlp/blob/main/data/UD/Acontecimientos_sobrenaturales.conllu&sa=D&source=editors&ust=1717607867023895&usg=AOvVaw3gzoN1EoQiznA9p15PbxWz)
			3. Try to recreate the same ML pipeline shown in the lambeq task above.
				1. Using spacy spanish tokenizer.
3. Concrete steps:
	1. Download code from the Lambeq tutorial’s   [Quantum](https://www.google.com/url?q=https://cqcl.github.io/lambeq/tutorials/trainer-quantum.html&sa=D&source=editors&ust=1717607867024391&usg=AOvVaw0hSG13wlkmHZP5akrEMoPD)  case
	2. Replace [training](https://www.google.com/url?q=https://github.com/CQCL/lambeq/blob/main/docs/examples/datasets/rp_train_data.txt&sa=D&source=editors&ust=1717607867024688&usg=AOvVaw0C4Tf2Ane5m0bQGPVI7Mj4)  data from relative clauses to the text classification task (see ‘Classical Case’ [https://cqcl.github.io/lambeq/tutorials/trainer-classical.html](https://www.google.com/url?q=https://cqcl.github.io/lambeq/tutorials/trainer-classical.html&sa=D&source=editors&ust=1717607867024954&usg=AOvVaw0iw7KHxbXYMsbUCXhl0ft6)
	3. Run the code and assess performance.
	
	4. Lambeq- removed “glue” words, or ones that don’t add to the semantics of a sentence. these correspond to the stop words of a language
	5. Colab notebook: [https://colab.research.google.com/drive/1krT2ibzrfLxin6VT5-nyXWr8\_HGEHNVF?usp=sharing](https://www.google.com/url?q=https://colab.research.google.com/drive/1krT2ibzrfLxin6VT5-nyXWr8_HGEHNVF?usp%3Dsharing&sa=D&source=editors&ust=1717607867025382&usg=AOvVaw3m2tOfqrp4UBXV1cY_-H13)
	6. research ansatz
		1. Advantage- it is able to better capture the richness of meaning
	7. Current status- spanish spacy tokenizer is able to tokenize correctly for Spanish. However, QBIT assignment is not working.
		1. How do we initialise the qbits?

8. Pipeline for the task:
	1. Use the same task as tutorial: [https://cqcl.github.io/lambeq/tutorials/trainer-quantum.html](https://www.google.com/url?q=https://cqcl.github.io/lambeq/tutorials/trainer-quantum.html&sa=D&source=editors&ust=1717607867025982&usg=AOvVaw2jfGhkUkfrlNbkzWkBIR2t)
	2. data- 100 spanish sentences from the uspantecan corpus
	3. ToDo- plan a classification task for uspantecan
	4. Robert Henderson’s repo: [https://github.com/bkeej/usp\_qnlp](https://www.google.com/url?q=https://github.com/bkeej/usp_qnlp&sa=D&source=editors&ust=1717607867026388&usg=AOvVaw0W2XkIkcQV3xFiThDTApyS)
	5. We select 2 data files, with different topics
	6. We create a classification task for differentiating between sentences under each topic

## Mar 19th, 2024

* What is [DisCoCat](https://www.google.com/url?q=https://cqcl.github.io/lambeq/glossary.html%23term-DisCoCat&sa=D&source=editors&ust=1717607867026871&usg=AOvVaw1JT_SF8YpM08KtOmjl2vs9) ?
	- Discrete mathematical category
* How do we choose a parser?
	- BobCat parser is the most powerful, so that’s the one bing used.
* Spider reader- we are not trying to build a bag of worlds model. We want to keep the grammar
	- It is faithful to lambeq
* Fix error with unknown word handling
* Monoidal structure-
* Start code for Spanish
	- Can bobcat work with Spanish?
	- how will PoS
* Step 3 Parameterization-
* Import Robert’s data
* ToDo- how to collaborate on a jupyter notebook cyverse
* ToDo- compile their code locally- edit SpacyTokeniser to change language from English to Spanish. Mithun is discussing this with the lambeq team
* ToDo- set up the dataset from Github to Cyverse, and use spanish translations.

# Mar 12th, 2024

* 1998 Lambeks paper on math===language: [here](https://www.google.com/url?q=https://drive.google.com/file/d/1BWhs5zOoA2n7y8aUnKoamfift0t9Xdhu/view?usp%3Dsharing&sa=D&source=editors&ust=1717607867028080&usg=AOvVaw1KcGOV2dbqQGcyH1m-yBwy)
* 2020 bob coecke [QNLP](https://www.google.com/url?q=https://drive.google.com/file/d/15hXA_ecFN31JJdt9E8POdUFT1mlcwssv/view?usp%3Dsharing&sa=D&source=editors&ust=1717607867028363&usg=AOvVaw0YfZ4zJY2KeDRyG4ZxWA1c)
* Type grammar revisited: [https://link.springer.com/chapter/10.1007/3-540-48975-4\_1](https://www.google.com/url?q=https://link.springer.com/chapter/10.1007/3-540-48975-4_1&sa=D&source=editors&ust=1717607867028593&usg=AOvVaw3TiubEsWFejcv65hog7NXV)
* [https://cqcl.github.io/lambeq/](https://www.google.com/url?q=https://cqcl.github.io/lambeq/&sa=D&source=editors&ust=1717607867028793&usg=AOvVaw2RpORwY67SCqRm2u84fRef)
* Neural networks use stochastic gradient descent- a top down approach. Language needs a bottom up approach
* Category theory hierarchy- everything is a category.
* Lambek- The Mathematics of Sentence Structure
	- Language is a bag of things, with 3 things- noun, sentence, NP. Anything can be created from these three
* Type Grammar revisited (Lambek 1999)
	- groups and proto groups- when an operation is done on members of a group, it remains in the same group?
* Combinatory Categorical Grammar
* Qubit- every time a decision is made, multiple options are collapsed into one. Until a decision is made, all possibilities are true. A qubit hangs out in infinite space, until it is acted upon by an operator.
* Quantum model for NLP Coecke
	- one grammatical category acting on a word, to move it one way or another
	- Dependency parsing looks like matrix multiplication
* Bag of words
	- sentences split into smaller meaning carrying chunks (words), which can be interchangeably combined in different ways
	- However- word combination is governed by semantic relationships between words
* Lambeq- pip install lambeq  to install
* If we know the minima of the gradient descent- can we build language up from it?
* TODO- install lambeq  and feed it a sentence \[done on Cyverse\]
* Run end to end- work on it like a tutorial
* Think- tokenizer available for English, Spanish, but not other languages. How do we work without one?
	- Run this on Spanish first
	- Think of a problem
* Jupyter notebook stored at: /data-store/iplant/home/mkrishnaswamy/qnlp
