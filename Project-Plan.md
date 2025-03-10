some definitions/possible experiment run combinations 

- Classical 1= the combination of (Spider parser, spider ansatz, pytorch model, pytorchtrainer) 

- Classical 2= the combination of (bobCatParser , spider ansatz, pytorch model, pytorchtrainer)

- Quantum1= (IQPansatz+TKetmodel+Quantum Trainer+ bob cat parser)- this runs on a simulation f a quantum computer

- Quantum2 = Quantum2-simulation-actual quantum computer=(penny lane model, bob cat parser, iqp ansatz, pytorchtrainer)


# References and links:
* [Google sheet](https://docs.google.com/spreadsheets/d/1NBINiUsAdrqoO50y_CX_BGGgXcP9Zt6i5nYKvuB70Tg/edit?usp=sharing) listing the latest status of all experiments at any given point of time
* live or dead status of all experiments as of :dec 10th 2024
	*  sst2_classical_1: (alive)
 		- runs well end to end+ has pytest. Latest version can be found in branch titled: run_sst1_classical1
	* sst2_classical2: (dead)
 		- i.e with bob cat parser, is hitting `both inputs must be same dtype` error again. end of road for now.Latest version can be found in branch titled: run_sst_classical2
	* sst2_quantum1:(dead)
   		- stuck on time out/ first .fit doesnt respond after a long time. Tried adding a pytest, but not able to cleanly capture a time out.
	* sst2_quantum2: (dead)
 		- same stuck on time out/ first .fit doesnt respond after a long time. Tried adding a pytest, but not able to cleanly capture a time out.
   
	* spanish_classical_1: 
	* spanish_classical2:
	* spanish_quantum1:
	* spanish_quantum2:
   
	* uspantek_classical_1: (alive)
   		- works+ have added a pytest
	* uspantek_classical2: (alive)
		- works+ have added a pytest
	* uspantek_quantum1: (dead)
 		- error (KeyError: ResultHandle('d06e4535-1edc-4190-b0e3-838b257d7612', 1, 10, 'null'))+added pytestx2
	* uspantek_quantum2: (dead)
		- error (Attributeerror:    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")AttributeError: 'PennyLaneModel' object has no attribute '_clear_predictions))+added pytestx2
  
 * latest command line command to run the code of any branch will be found next to its respective test case
 * always run pytest before merging a branch or even git push
  
* Link to github started by Robert Henderson:
	* Private repo with the QNLP dataset, shared by Robert Henderson: [https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data](https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data)
	* QNLP code repo [this repo]: [https://github.com/ua-datalab/QNLP/blob/main](https://github.com/ua-datalab/QNLP/blob/main)
* Link to white paper: [https://www.overleaf.com/4483532232tcfnfdrrcbdc#12a1b4](https://www.google.com/url?q=https://www.overleaf.com/4483532232tcfnfdrrcbdc%2312a1b4&sa=D&source=editors&ust=1717607867015283&usg=AOvVaw0VwgWn_tu2jNMuTmaj2PDL)
* All data (e.g. spanish only files) is stored in a [gdrive folder here](https://www.google.com/url?q=https://drive.google.com/drive/folders/1m4nFZwsUcZ2DQzN3nYaK0_oKJXGhV575?usp%3Ddrive_link&sa=D&source=editors&ust=1717607867015673&usg=AOvVaw32Cbwsxm70wOGxbbRLFbb0)
	- Uspantekan data: [https://drive.google.com/drive/folders/1CtMhTf-v0nSUSaTJVelILkDMrLfF1U5Y?usp=share_link](https://drive.google.com/drive/folders/1CtMhTf-v0nSUSaTJVelILkDMrLfF1U5Y?usp=share_link)
 	- Spanish data: [https://drive.google.com/drive/folders/1SThJ6tyUAzvfVSFo6w_VyB4HPt381jp1?usp=share_link](https://drive.google.com/drive/folders/1SThJ6tyUAzvfVSFo6w_VyB4HPt381jp1?usp=share_link) 
* Jira Link: [https://cyverse.atlassian.net/jira/software/projects/QNLP/boards/27](https://www.google.com/url?q=https://cyverse.atlassian.net/jira/software/projects/QNLP/boards/27&sa=D&source=editors&ust=1717607867016357&usg=AOvVaw2fccm9pIgF5Yw5sAb26eH0)     
* [Miro Whiteboard](https://miro.com/app/board/uXjVKVPCIK4=/?share_link_id=77584526552) 
* Papers:
	* the most fundamental paper which introduces QNLP is 2010 DISCOCAT [paper](https://drive.google.com/file/d/1T7H5WH1q0mKng-zwqOYrlqEkBpOIcUDR/view?usp=sharing)
 	* To undrestand that, you need to understand 1998 Lambeks paper on math===language: [here](https://drive.google.com/file/d/1WmHNND7geQTfO3sRK-NDoOBHtKZL7pAa/view?usp=sharing)
  	* To understand 1998 lambek's work, you need to understand [1958](https://drive.google.com/file/d/1mXmLMH9NbQgrbB550XIaOph07yRV3INO/view?usp=sharing) lambek's work 	
	* 2020 bob coecke [QNLP](https://www.google.com/url?q=https://drive.google.com/file/d/15hXA_ecFN31JJdt9E8POdUFT1mlcwssv/view?usp%3Dsharing&sa=D&source=editors&ust=1717607867017007&usg=AOvVaw2jRw8msgoQEVE_z5vZxQCa)
	* Type grammar revisited: [https://link.springer.com/chapter/10.1007/3-540-48975-4\_1](https://www.google.com/url?q=https://link.springer.com/chapter/10.1007/3-540-48975-4_1&sa=D&source=editors&ust=1717607867017296&usg=AOvVaw0T99YvALpGGqp50dAnYxz9)
	* Khatri et al. thesis: [https://drive.google.com/file/d/141UTEmduZoFhq0-d1peWLGdUhMLpNlho/view?usp=share_link](https://drive.google.com/file/d/141UTEmduZoFhq0-d1peWLGdUhMLpNlho/view?usp=share_link)
* Colab notebooks:
	- Data pipeline: [https://colab.research.google.com/drive/1YwdVkFZRt30QPuUwQ-y9W1vSnYlkS656?usp=sharing](https://www.google.com/url?q=https://colab.research.google.com/drive/1YwdVkFZRt30QPuUwQ-y9W1vSnYlkS656?usp%3Dsharing&sa=D&source=editors&ust=1717607867017671&usg=AOvVaw3sePnYQ_2mwLcqo1YYvu9Y)
	- Lambeq for Spanish, run with Spider parser: [https://drive.google.com/file/d/1wTo8rAObpuLu65DyFo1D0gE5kKjUtzBf/view?usp=sharing](https://www.google.com/url?q=https://drive.google.com/file/d/1wTo8rAObpuLu65DyFo1D0gE5kKjUtzBf/view?usp%3Dsharing&sa=D&source=editors&ust=1717607867017990&usg=AOvVaw1oMypNSQtjg_K-olMfxRnv)
 	- Khatri et al, for Spanish: [https://github.com/ua-datalab/QNLP/blob/megh_dev/OOV_MRPC_paraphrase_task.ipynb](https://github.com/ua-datalab/QNLP/blob/megh_dev/OOV_MRPC_paraphrase_task.ipynb)
  	- Khatri et al, for Uspantekan: [https://github.com/ua-datalab/QNLP/blob/mithun_dev/v2_khatri_thesis_version_which_gave1_mnli_100_runs_end_to_end.ipynb](https://github.com/ua-datalab/QNLP/blob/mithun_dev/v2_khatri_thesis_version_which_gave1_mnli_100_runs_end_to_end.ipynb)

# Working Group Best Practices
- Golden rule: Don't get emotionally entangled with results/errors. They are just that, problems to solve. Don't take them personally/nothing to do with you.
- Never work on main branch (unless you are working on Projectplan.md)
- After every code change, before pushing, always run pytest before the actual code in any branch. This can be achieved using `./runner.sh`
- For every new bug fix/new feature create a new branch.
- Always merge and test that new branch with `staging_merge_here_and_test_before_merging_to_main`, checkout on laptop, run pytest
- merge `staging_merge_here_and_test_before_merging_to_main` with `main` only if it clears all pytests 
- Save all data in the private repository, to prevent leaks: [https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data](https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data)
- Save code to public repository, so it can be opened on Cyverse and run on colab: [https://github.com/ua-datalab/QNLP/blob/main/](https://github.com/ua-datalab/QNLP/blob/main/OOV_MRPC_paraphrase_task.ipynb)
- Cyverse has resource allocations- so all big training done there. Example: 

# Meeting Notes

## Dec 16th 20204
- Pytests for each specific usecase:
- BobCat parser is not working
- Quantum case is not functional
- Chaking if code works on Cyverse

## Dec 12th 2024
todos:
- overall: add pytests: no fixing things. just add pytest
- add pytest for  run_sst1_run_sst1_quantum1 ---done
	- add time out 
 	- add if wandb
  	- merge to staging
   	- merge to main
- add pytest for run_sst1_run_sst1_quantum2---done/ignored
	- mostly same as above- timeout
- make wandb optional  ---done
	- in a branch called make_wandb_optional_commandline_arg ---done
 	-  run pytest locally --done  	
  	- merge to main **---done
  	- pull and run pytest locally : started at 7pm dec 12th
- create 8 pytests for spanish
	- classical1 x2
 	- classical2 x2
   	- quantum1 x2
    	- quantum2 x2
- create 8 pytests for uspantek
	- classical1 x2**---done**
 	- classical2 x2**---done**
   	- quantum1 x2
     	- quantum2 x2
- add python dictionary
- start tuning uspantek with whatever of the 8 above you think is best 
 

## Dec 11th 2024
- Discussion of monthly goals and future plan:
	- English: find a dataset for topic classification, which is an NLP task that Khatri et al. did not run in their work.
 		- Q: Can QNLP offer flexibility in NLP task, or is it restricted to one task alone (text paraphrasing)
	- Spanish, Uspantekan: actually set benchmark for topic classification. English results will offer further evidence    
- updates on `runner.sh`: it runs `pytest` first, to ensure the code updates do not break existing code.
- Question

Megh todo
- Setup and Run english pipeline on cyverse, end to end: assess and document
- Update readme with all steps for running the pipeline
- Paper

Mithun todo
- Make wandb an optional parameter in argparse, (save metrics to csv?)
- Push updated `run_me_first.sh` to `main` branch ---done

## dec 10th
1. add pytest for run_sst1_run_sst1_classical2 for yes expose- catch error---doe
2. add pytest for run_sst1_run_sst1_classical2 for no expose ---done
3. add pytest for  run_sst1_run_sst1_quantum1
4. add pytest for run_sst1_run_sst1_quantum1
   
## Dec 9th
- Pytest implementation complete
- Code is functional and has checks. 

## Dec 8th 
Todo at EOD Dec 8th:
1. merge branch `run_sst1_classical1` with `main` and delete branch
	2. update: merge done. Branch not deleted. Will stay back to hold latest version of god
3. make yes and no for `if expose val` a command line argument `action=store_true` - branch for itself? ---done
4.  add pytest for run_sst1_run_sst1_classical1 for yes expose---done
5.  add pytest for run_sst1_run_sst1_classical1 for no expose ---done

### Mithun coding
How to run code without debugging but using command line arguments:

`python classify.py --dataset sst2 --parser Spider --ansatz SpiderAnsatz --model PytorchModel --trainer PytorchTrainer --expose_model1_val_during_model_initialization False`
How to debug from command line.
1. add these code to the top of your classify.py
```
import debugpy
debugpy.listen(5678)
print("waiting for debugger")
debugpy.wait_for_client()
print("attached")

```
2. view->terminal
3. python classify.py --dataset sst2 --parser Spider --ansatz SpiderAnsatz --model PytorchModel --trainer PytorchTrainer
4. while its waiting click on the play button on the left most column panel with a bug on it.
5. if(launch.json) is not there
	6. create launch.json
	7. pick python debugger
	8. pick remote attach
	9. pick localhost
	10. pick port: 5678
11. next launch.json with the following text will be created
    ```
    {
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ]
        }
    ]
    }
    ```
    11. now just click the play button next to top left Run an: Python Debugger:Remote attach (or whatever name you choose to provide in the luanch.json)
    12. this is extremely useful when you have command line arguments to pass in inside vscode, and yous till want to debug line by line. Else there is no other way to get the command line arguments, required ones
    
## Dec 4th 2024

### Update @10pm
- with max tokens per sent =10
- sst2 
	- also using only 20 in train, 10 in val and 10 in test. 
	- classical 1 (spider parser): runs
 	- clasical 2 (bob cat parser): dtype error- no val data after max len=10
  	- quantum 1: error (atleast its not out of memory error) ((max token length=12 because else val was becoming empty)
  	- quantum 2:taking forever+ have to run overnight(max token length=10 because else val was becoming empty)
- uspantek
	- classical 1: runs
 	- clasical 2: runs
  	- quantum 1:
  	- quantum 2:
	 
- spanish
	- classical 1: runs
 	- clasical 2: runs
  	- quantum 1:
  	- quantum 2:

- Hackathon quest: Spider parser runs without error, Bobcat throws errors. We need to find the differences between both parsers, in order to debug Bobcat, which is the real parser we need to implement.
- things we can do today
	1. Classical 1 - Uspantek- add dict and see if we can improve accuracy (want)
	2. Classical 2- Bobcat parser (need)
- bobcat parser works well on quantum2 and quantum1 pipeline.
- Question: why does bobcat parser give errors for things spider parser works fine for?
	- is it sentences2diagram? Update: nope. Doesnt matter if we are using sentence2diagram or sentences2diagram 
	- is it the none returned added when suppress_exceptions=True for bobcat parser? -Ans/Update: no, that didnt make a difference/same error.
 	- is it because during bobcatparser declaration we use 3 base types (N,NP,S)instead of the two original defined by lambek (N,S)-Ans/Update: no, that didnt make a difference/same error.
  	- what happens if we give no root_cats in bobcat declaration at all-Ans/Update: no, that didnt make a difference/same error.
  	- what happens if we add verbose =text in bobcat parser declaration-Ans/Update: no, that didnt make a difference/same error.
  	- update: doesnt error out for food_it data. Only sst, usp, and spanish..weird
  	- what happens if you combine bobcatparser+IQpansatz+pytorchtrainer+pytorchmodel- symbol has no attribute size. clearly bobcat+iqp needs quantum trainer and atleast a tket model
  	- what happens when you reduce the max token length to 10. ans:holy shit it works..for spanish...atleast passes first fit. there is a bug in secnod eval of model 4 weights.wow, token lenght was the issue? WTF bobcatparser-
  	- ## update. passes all 4 models and succesfully produces result for spanish in classical2 @10pm dec 4th 2024
  	- what is the max you can go without error-?ans: it is 10 , especially for a new language like spanish or uspantek WTF
  	- HOW ABOUT uspantek? ans: throws away shit load of sentences, but yes, passes fit 1/model 1. now hitting a bug in model3 np.zeroes
  	- how about for sst2/english?- that maa ka lavda tho is barfing even with max token size=10
  	- so will your quantum models 1 and 2 reply if you have tokens less than 10 for sst2?
  		 
  - will it work if we combined bobcat parser with some other ansatz e.g. tensor?.ans/update: hits another error because looks like tensor ansatz doesnt following andrea_n@s_0 format
  - - what are the typical ansatz that the sample code/examples in lambeq docs use? bobcat+spider, bobcat+IQP, Bobcat+IQP
    - meanwhile in main branch, turn on wandb and run 3 classical1s (sst, spanish, uspantek)
    - maybe stop doing top down knob turning/mixing and matching/permutation and combination and instead try reading and understanding the error more deeply
- Mithun implemented many try catches to investigate output at multiple points in the code   
 - Sentence2Diagram and Sentences2Diagram work differently
 	- of 20 sentences, 7 removed for colloquisms, fragment- parser couldn't work with them 
- Update: Unable to find source of main error
- ToDos for meg:
	-  set up argparse correctly to work with pytest, so we can run unit tests: right now, all requirements have been turned to false. We need to set things up so it works with required arguments
		- ToDo (pri5): set this up withlaunch.json on VSCode
 		- `logger.info()`: learn how to set errors, warning, info
 	- Set up pytest so everything runs end to end
 	- Thought: add a print statement as the first line of code, telling users how to run the script?
  	- Find a way to save/print all sentences that did not get parsed, so we can check for why that is

## Nov 30th 2024
### Mithun's Logs and meeting notes: tatus@Dec2nd202410am
- **English**
	- **SST**
	- **Classical 1** (Spider parser, spider ansatz, pytorch model, pytorchtrainer)
	- classical 1.a:(with 20 data points)
    		- Classical 1.a.1.- Without exposing val to the model
			- "status: Model 1: training accuracy (30 epochs): 92% val Accuracy: Not applicable
   			- model 3 (oov model + model 4: 60% accuracy)
		- classical 1.a.2: with exposing val to model
    			- "status: Model 1: training accuracy (30 epochs): 92% val Accuracy:65%
				- model 3 (oov model + model 4: 55% accuracy)
		- classical 1.b:(with 100 data points train +1000 test)- started at 3.19pm from terminal in the folder testing_megh_*
			- **Classical 1.b.1**.- Without exposing val to the model 
				- "status: Model 1: training accuracy (30 epochs): _____ val Accuracy: Not applicable
				- model 3 (oov model + model 4: ------- accuracy)
          		- **Classical 1.b.2**: with exposing val to model
          		- "status: Model 1: training accuracy (30 epochs): ----- val Accuracy:_____
      			- model 3 (oov model + model 4: _____% accuracy)
    		- Classical 1.c:(with 10k data points train)
    	- **Classical 2** (BobcatParser, spider ansatz, pytorch model, pytorchtrainer)
     		- Status: same DTYpe mismatch error
		- Possible solution: its a bobcat parser error. Bobcat parser doesnt like sentences more than a certain limit. usually its 9 or 10 tokens. So remove all sentences in dataset which is more than 10 tokens
		- Todos
			- run again and confirm
			- make sure it is ran over 65k training data
	- **Quantum1-simulation**  (IQPansatz+TKetmodel+Quantum Trainer+ bob cat parser)):
       		- status: ran into memory issues needs 460 Million GB memory or something
         	- possible solutions/next step: **end of road**
   	-  **Quantum2-actual quantum computer** (penny lane model, bob cat parser, iqp ansatz, pytorchtrainer):
		- No response for first fit for a few hours)
	      	- Possible solution: does that change if you expose val data during model creation?
           	- Run it on cyverse.  ?
   	- MNLI ?
   	- MRPC ?
   	- Food IT toy data
   	-  Classical 1 (bobcat parser, spider ansatz, pytorch model, pytorchtrainer)
		- status: 
		- possible solution: 
	- Classical2  (Spider parser, spider ansatz, pytorch model, pytorchtrainer)
		 - status: 
		- possible solution: 
	- Quantum 1 -simulation (IQPansatz+TKetmodel+Quantum Trainer+ bob cat parser)
		- status:
	- Quantum 2 -actual quantum computer (BobCatParser+PennyLaneModel+PyTorchTrainer+IQPAnsatz)
		- status:
- Non English
	- **Spanish**
 		- **Classical 1 (bobcat parser, spider ansatz, pytorch model, pytorchtrainer)**
			- **status: error mismatch datatype**Should be an easy fix.check branch spanish
   			- update: Turns out not so easy fix
       			- Also check if exposing val helps.--nope same error.
          		- Also will increeasing/reducing base dimension help- different erro,ValueError: not enough values to unpack (expected 2, got 1).
		- **Once you find out what exactly is base dimension it will solve your dtype issue also**
		- Note: am working on the new branch:` merge_spanish_to_main`
	  	- Use `suprress_exceptions=Tru`e--nope same error
	  	- Dont use 	`remove_cups`--- nope same error
			- Also build bobcat parser locally- there are some words which its giving dimension of 10- even in small sentences? i mean the word itself is n_ something. which means just one noun. then why 10? update, the word is NameError("name 'y_9__n' is not defined"). in sentence 15, eh tanto problema
      		- **Classical2 (Spider parser, spider ansatz, pytorch model, pytorchtrainer)**
        		 - `status:IndexError`: when no subspace is given, the number of index arrays cannot be above 31, but 32 index arrays found- if we do  if len(tokenized_sent)> 31: . If we do len(tokenized)>32 we get:
				- update used len(tokenized)>19 and it moved past first fit. wtf.
           		- update: **without exposing val during first fit. gives 95% training accuracy on first fit. bad news is OOV model3 accuracy is 54%**
		-  todo:
		a. **run with exposing val during first fit: after first fit, model1 training accuracy: 85.54 model1val accuracy = 77.27 model3 accuracy=45% (but that has early stopping from another dataset food_it/is tunable)**
		b. for both spanish 1 and 2 classical experiments, can we improve accuracy of model 3.
			- then find a standard benchmark in spanish
           		- in the absence of huge datasets, QNLP is a viable option.           	
		- **Goals**
			1. benchmark for QNLP + spanish
			2. see what happens if english is not the core language
				- Qn) why not LLMs
	           	  	- Ans: we think there are no natively trained LLM.  
	           	- (want)bring in GPT embeddings and see if accuracy improves
	          	- is it scaleable on- spideransatz- if yes: try on large spanish dataset? say 10K sentences
	          	- has there someone trained an LLM in spanish from scratch- if yes what dataset,. how much can we get.           	
		- **Quantum 1 -simulation (BobcatParser+IQPansatz+TKetmodel+QuantumTrainer)**
			- status: Taking a long time, buut mostly will be not enough memory
			- possible solution: **Try to run for hours, if memory issue, end of road**
		- **Quantum 2 -actual quantum computer** (BobCatParser+IQPAnsatz+PennyLaneModel+PyTorchTrainer)
			- status: killed- taking a long time
			- possible solution: **todo:Try to run for hours- maybe from cyverse?**
	- **Uspantek**
 		- Classical 1 (Spider parser, spider ansatz, pytorch model, pytorchtrainer) + exposing val data
			- status: **runs end to end. phew. model 1. train accuracy =90% validation accuracy: 59.09 (note: this is 30 epochs- which is hard coded. tuneable)**
				- model 3  accuracy=50%
				- todo: run without exposing val data
   	      			- model 1 training accuracy 90%
   	         		-  model 3: 54.54%
			-  todo: value after adding in word alignment	
   	  		
      		- Classical2  (bobcat parser, spider ansatz, pytorch model, pytorchtrainer)
        		- status: **both inputsshould have same dtype**
      		 	- possible solution: **Should be an easy fix**. check branch spanish
   	      - todo
   	      	- Alignment dictionary translate basedon the dictionary between uspantek and spanish   	       	    			
           	- Quantum 1: simulation (IQPansatz+TKetmodel+Quantum Trainer+ bob cat parser)
		 	- status: Taking a long time, buut mostly will be not enough memory
    			- possible solution: **Try to run for hours, if memory issue, end of road**
    		- Quantum 2: actual quantum computer (BobCatParser+PennyLaneModel+PyTorchTrainer+IQPAnsatz)
		 	- status: IBM cloud login issue

## Nov 29th 2024
- Continuing with SST2 data
	- getting ty(p) error. Update: fixed it by adding p to ansatz
	- next: AttributeError: 'int' object has no attribute 'rotate: 
		- try replacing spider ansatz with tensor ansatz
		- update. no luck, same error. Not sure what the error means, what has 
	functor got to do with english data. Todo: try plotting diagrams and circuits of first data point
		- update : diagram of first data point looks ok. todo: switch back to spider ansatz, to avoid too many moving parts ---done
	- todo: put try catch around erroring out circuits update: that worked. however out of initial 80 train data points, now its only 49 circuits. todo: figure out what is going on.
	- next error in .fit()  both inputs should have same dtype. todo: switch to everything quantum. if we are fighting might as well fight in quantum world
	- update: using IQPAnsatz, TKetmodel, Quantum trainer. now getting: ERROR: Insufficient memory to run circuit circuit-166 using the statevector simulator. Required memory: 4398046511104M, max memory: 16384M todo: a) try removing cups. else b) call onto penny lane or actual quantum computer?
		- open up the cup removal--done
	- update: cup removal was useful. for example in the diagram of the first sentence, before cup removal tehre were 14 units, after removal it became 7. But still hitting memory issues.needs 4.29 Million GB of RAM...that too just for 10 sentences. beautiful..with remove cups it might come to 2million GB. very helpful
		- try replacing spider ansatz with tensor ansatz
	- after runs end to end 
		- run pytest for food_it (pass variables) ---done
		- add pytest for sst2
		- turn on tuning
		- get values with and without exposing dev during initialization
		- move to cyverse for big run
		- start replacing with Quantum trainers, iqp ansatz etc
- Mithuns log
	- branch. Stabilize_pytest_driver
	- to create
		- driver function into which you pass 4 variables
 	- add a 5th variable, exposure val during build+ create new pytest case

## Nov 28th 2024
- Dataset research: GLUE may not be a good fit for current setup. Need to find a classification task
- VS Code debuggers: ran tests and Mithun rechecked conda installations on a new environment for Megh, with a functioning debugger calling the right environment.  

## Nov 27th 2024
- todo: load english datasets
- start with sst2- sentiment in glue- branch called read_sst
	- read dataset into our code
 	- run our base code end to end
  		- hit bobcat parser not able to read some sentences, solution: put try catch around them
    	 	- next issue: laptop is maxing out. its showing 3 hours for parsing spider diagrams. most likely will hit ram max also. solution: move to cyverse
       	- - update: laptop managed to run training on 65K documents. but now hitting ansatz related some key error. Taking a pause for thanks giving break. This is becoming a very unhealthy addiction of - maybe next run will give me the big ground breaking/life saving result. Like megh says Very bad idea to put all eggs in same basket/unhealthy getting emotionally entangled with technology and inventions.
  - spanish: branch called spanish_experiments
  	- run end to end
   		- hit bobcat parser not able to read some sentences, solution: put try catch around them
     		- next issue Ty(p) not found. This means, there are prepositions, So both ansatz and bobcat parser need to be initialized with them
       		- still getting some domain codomain issue: pausing for now.  havent merged branch to main. Merge only if code runs end to end and you are completley confident nothing will break

## Nov 26th 2024
- Updated code walkthrough
	- All code in `classify.py`, without Mithun's notes, just tags. Copy of `v7` code.
 	- Old code in archive. In archive, `no_pair...` is for classification task, `yes_pair...` is Khatri's code replication, which we can't use because of depriciate d packages.
  - New branch: `tuning model3 again`: new branch, modularized for specific upgrades.
  	- Tuner for model 3 in the code (see below).
   	- Performing a grid search, with optimizers, loss functions, batch size, etc. to find the best fit.
    	- Replaced RandomSearch with gridSearch
    	- Functions for permutations and combinations with hyperparameters. Stuck at 83% accuracy
- Sanity check to make sure code works correctly completed.
- Upcoming steps: move away from toy data, and find a real-case dataset.
- ToDo for today: set up F1 scores for all of the code. Found implementation for pytorch
	- bug: F1 function ran through tuner, which slows things down due to grid search
 	- Implementing F1 scores for model
		- Scores fell dramatically when `if tuning` feature added. Issue in how the code is set up with continuous integration.
	- Fix: removed randomly assigned weights implemented for demo yesterdday 
- todo: load english datasets
- start with sst2- sentiment in glue
	- read dataset into our code --done. is in a branch called: read_sst
 	- run our base code end to end
  	- if runs without any error, merge to main
  	- else if its Hadware limitation, move to cyverse
  - tuning
  	- search for the word todo in classify.py
## Nov 26th 2024
- Demo and discussion with enrique:
	- Need real-world datasets, not toy datasets.
 	- It's ok to expose model to add data, so long as model doesn't learn from it.

## Nov 20th 2024
- Mini hackathon for setting up Uspantekan demo
	- Going back to [code from Spe 4](https://github.com/ua-datalab/QNLP/blob/abbe80fd0d5a40f8920505c68780a9f57f76d8cc/v7_merging_best_of_both_v6_andv4) (last date of demo)
 	- Accessing previous parameters and settings, so that we can upgrade the code correctly.
  		- Hardcoding of circuit dimensions for noun and verbs- good for saving GPU useage, very bad for flexibility.
    			- Most recent updates can't run older code and process Uspantekan data: we will need to overhaul this. This is because the dimensions are hardcoded, and we changed the Ansatz and other pre-requisites.
- Spanish, with Spider parameters
	- model is training, loss going down, just very slow. Val performance around 45-51%.
 	- No early stopping: needs val and dev yet, since we are only running train data. But once val dataset can be run with fasttext embeddings, we will be able to add early stopping.
- ToDo: replace `split()` with regex, for added flexibility.
- Trying to load uspantek
- Updates:
	- end-to-end model with Bobcat parser for English ready- all that we need is a thorough code review, and complete switch to PyTorch
 	- We are able to run it with just train data. 
 	- Spanish and Uspantekan: input data not cleaned right, more OOV words than English (predictable). Lots of '\\' symbols in the words
  		- Over 30 OOV words.
- Todo
	- open up remove cups writer if bobcat parser is used. test for english first.
	- spider ansatz raise a pull request with LAMBEQ guys- for format of symbol spider does aldea_0__s while everything else does aldea_s_0
 - Todo: find out how to add early stopping. in model 1
 - inside run_expt:
 	-  why is he setting random seed, that tooin tensor flow especially since am using a pytorch model.tf.random.set_seed(tf_seed)
  	- both lists tf_seeds  and nl should have more than 1 values- do only on HPC or cyverse though
Trying to load uspantek
- Todo
	-  clean up accents/utf-8 on both spanish and uspantek?
	-  in the function generate_initial-*
		-  there is one if(ansatz_to_use==SpiderAnsatz): but no else
		- for initial weight vector , instead of blindly taking the first 2 or 4 cells from the fasttext embedding, initialize with something more meaning ful. say sum of all cells?
 	 	- what is the connection between initial param vectors and labels? in QNLP labels directly are logits/confidence in predictions?
	- while initializing ansatz:2) Noun should have a higher dimension than sentence? how? 
		- it will be really cool enhancement if we can have a combined training of model1 and model 4, with the embeddings being updated on the fly.- and not training model4 offline after model 1
	- open up remove cups writer if bobcat parser is used. test for english first.
	- run model 3 to 100% accuracy- i.e dont do early stopping unless accuracy has crossed 100
	- note, early stopping should be done on the val data- otherwise training loss is always going to keep changing/decreasing/overfitting. what is our dev in model 3?
 - update: didnt og anywhere. Pushed in a separate branch
 - after 7pm coding:
	- model1 Sanity check
    	- why stop at 30 epochs.
    	- if val data is provided, can we implement early_stopping inside the code itself
    	- why not even provide a hardcoded val_v2 data, just for early_stopping checking. Maybe the last check can be on testing 
  	- model 3- Sanity check
   	- Make sure the OOV code is completely working.
    	- Check how early stopping ka dev is done
    	- 3. when we show the val circuit to model 1, we get 98% for model 4- why not 100%,
    		- find if early stopping is stopping too early
    		- also take a word which exists both in train and val and see how much is the weights difference, i.e when using model1 ka real training vs model 3 ka prediction. paste the results below here
  	- Tune model 3 to the maximum so that you get 100% on model1’s dev data. THings to tune can be
	- USE IN BUILT KERAS [TUNER](https://keras.io/guides/keras_tuner/getting_started/)
 		- all types of activations
	     	- all range of learning rates
	    	- all types of optimizers like Adam,SGD
	     	-  OTHER LOSS FUNCTIONS like binary_crossentropy
    	- no of layers    
    	- Why only two layers
    	- Encoder decoder?
    	- Try pytorchtuner for first model.fit()
Code clean up: keep a copy of the code with inline comments, and create a version without any comments.

## nov 19th 2024
1. todo: ask enrique why assigning embedding values to the first model is making the model stuck/not reducing loss- this is model1 qnlp- update this might be requires_grad
2. find how to add early stopping to the model 1s training
3. when we show the val circuit to model 1, we get 98% for model 4- why not 100%, - find if early stopping is stopping too early
4. when we dont show the val_cicruit- we get 83% when we showed the val_circuit - with one new symbol 98%- now try asking chatgpt to find more sentences with more oov for val and see how our model does. in both scenarios above. with and without showing val during initialization of model.
5. keep a copy of the code with inline comments, and create a version without any comments.
    
## Nov 18th 2024
- Runthrough of current code
	- For English, current performance with Fembeddings and Bobcat parser is 82%
 	- Found English fasttext `.bin` file: it lived in another repository
  	- Unable to replicate Khatri et al., as the parser he uses no longer works for new OSes, but we have all the resources to run a similar system
  	- Assessed qbit requirements for each part of the sentence: found in max_param_length. This variable is used to create an empty vector of that size, to avoid size mismatch between vectors
  		-  Mithun changed the base dimension from 1 to a variable number, to account for transitivity in verbs instead of assuming the dimensions.
- Mithun's updates:
	- Made a big break through yesterday night. Atleast I think so.
		- their lambeq code was initializing the QNLP model with train and val circuits
 		- i initialized just with just training circuits
  		- but then i noticed our loss values were not decreasing (and the model weights not being updated) in the first .fit()/training of the QNLP model
   		- I went digging and realized, that it is the generate_initial_param thing Khatri does which is screwing up things. I still am not sure what that function does. We are already initializing model1 with random values. anyway, the moment i commented that out,model 1 loss started dropping and hit like 99% accuracy. Even better, model 4( the prediction version of model 1 combined with values of OOV)- gave 82% accuracy. Now IMHO that is huge. i.e a model not seeing val data, training only on 70 sentences, vs 30 in val. 
    	- todo confirm that the flow is right and this is not a fluke
    	- update: looks good to me. If my hunch is right ***this result Is a paper in itself*** update@nov20th2024. Dissecting with megh acting as devil's advocate ongoing
	- QN) so the ideal flow of events during prediction should be, we go thro ugh each word in val, check if it exists in train vocab, if yes get its already trained weights from model1, i,e the first qnlp model. IF NOT then go get the corresponding embeddings from fasttext, give it to model3, which will output a new weight vector for you, which then you attach to the prediction model, i.e model4, saying this is the missing piece. Todo: confirm if this is how khatri is doing it  ***--done***
		- Ans: No. He is taking every word in val, giving its embedding to model3 i.e the OOV model, which then gives its weights, and which he is using. I mean, ideally the weights that the OOV model predicts for the val word must be same or very close what the model1 had learned. ...but either way,
  		- update@nov 20th: found that he is using dict.get() . i.e his logic is same as what we mentioned above. i.e he gets the word's weights from modlel1 ka trained weights if it exists. IF NOT THEN ONLY does he go to embedding space and model 3. Brilliant 
	- this is a nice todo:
		- run experiment with our flow chart idea and see if that changes anything.
		- also take a word which exists both in train and val and see how much is the weights difference, i.e when using model1 ka real training vs model 3 ka prediction. paste the results below here
 	- also note, i had deleted the mithun_dev branch upstream while my laptop still thinks it exists. create a new branch asap locally and push to remote. else all the changes and commits will be lost - do the nasty way of cp -r and new folder for now  ***--done***

## Nov 17th
- todo from yesterday; start experiments, especially with bobcat and classification.
	- for no pair:
 		- add unit tests to CI on github ***--done***
   			- update: done till fasttext model loading issue.
      			- cache fasttext model so that you dont have to download it every time
         		- update. can't cache fasttext during continous integrations since its a fresh ubuntu virtual machine every time.  
   		
 		- move wandb to no pair file. **---done**
   			- add parameters  in wandb ***--done***
			- separate out dev and train epochs variable names ***--done***
   			- turn on wandb and ensure you can see them on web browser ***--done***
 		-  check if there are any other features i added in yes pair file in the last one week, if yes move to no-pair ***--done***
   		- use english fasttext embeddings **---done**
     		- create a variable dataset to use- and add it to arch and then into wandbparams. ***--done***
       		-  add unit tests **---done**
         	-  - move dev epochs count to wandb param **---done**
          	- add type of data also to wandb arch **---done**
          	- - map out possible combinations in spreadsheet --done. rather building it (here)[https://docs.google.com/spreadsheets/d/1w6u7xbR3Q37fh8uhgIJw230yWQetXdrZlvK42msIy80/edit?usp=sharing]
          	- change expected value in test file based on new config. eg. english vs spanish embeddings ***--done***
          	- add early stopping to training data 	
       		- move to cyverse         	
          	- increase the food it dataset to max size- currently seems to be only 18 in training. should be close to 100
          	- create a main function so that pytest doesnt have to call it twice.
          	- go through every single line of code in yes pair manually and check if it has been ported to no pair code
	-  for yes pair
 	-  how did I get MNLI==1 - that code, how did it cross the depccg parser issue? and the ansatz comparator issue?

- planning for experiments:
	- Qn) Did you add test cases?
		- Ans: yes. test_oov_no_pair.py. Can be run using the command `pytest` on command line.
	 	-  todo: either add this to a start up file, or to continous integration on github
	- Qn) What was the accuracy on dev set when you ran Food IT code +oov with spanish embeddings
		- Ans:43.3%
	- Qn)Did you try running food IT code with english fast text embeddings.
		- Ans: No not yet.
		- update: just did. accuracy in dev is  51.66%. As much as this is not great, but this 8 points bump is huge IMHO, which means the patient is alive and is responding to meds.
	 - - Qn) Remind me, what exactly are we using OOV for?
		- Ans: So in FOOD_IT, there was exactly one symbol (not word) that was out of vocabulary. it was person_n@n.l or something. So when we test using the model that was 
	- Qn) What is the configuration you are currently using for FOOD-IT?
		- Ans:
	 		- parser_to_use = bobCatParser  #[bobCatParser, spiders_reader]
			- ansatz_to_use = SpiderAnsatz #[IQP, Sim14, Sim15Ansatz,TensorAnsatz ]
			- model_to_use  =  PytorchModel #[numpy, pytorch]
			- trainer_to_use= PytorchTrainer #[PytorchTrainer, QuantumTrainer]
			- embedding_model_to_use = "english"
	- Qn) what is the plan forward?
		- Ans: High level: slowly bring in quantum stuff. Especially a quantum ansatz
	- Qn) What is the detailed level plan forward.
		- Ans:

## Nov 16th
### Mithun working on reproducing khatri values
"stopping/giving up on trying to reproduce khatri code. Main issue is he uses depccg parser, which is impossible to setup. I tried bobcat parser, and spiders reader and even tree reader. bobcat parser, doesnt do well with remove cups, spider and tree does, but then they hit the ansatz error saying circuits vs diagrams.its amess. time to call it quits"
- todo: update code with code for 1 sent (no pair)
- update: done. we have 2 files now, one for yes pair of sentences (OOV_classification_yes_pair_sent.py) and other for nopair -classifcation of food IT(OOV_classification_no_pair_sents.py)
- todo; start experiments, especially with bobcat and classification.
	- for no pair:
 		- move wandb to no pair file
 		-  check if there are any other features i added in yes pair file in the last one week, if yes move to no-pair
   		- use english fasttext embeddings
     		- add unit tests
       		- add unit tests to CI on github
       		- move to cyverse
         	- map out possible combinations in spreadsheet
	-  for yes pair
 	-  how did I get MNLI==1 - that code, how did it cross the depccg parser issue? and the ansatz comparator issue?
- everythign is in mithun_dev branch as of today

## Nov 14th-
### Mithun working on reproducing khatri values
- using just one file OOV_classification now on+ passing arguments as to pair based dataset or not. THe pair no pair file difference was causing too many versions.

## Nov 13th
1. megh pointed out that even thoughy i have end to end code OOV for food-IT, it is not correct, because I am using spanish embeddings. So as of now, we DONT have any end to end code. I was using spanish embeddings for english. We tried giving spanish data, and teh parser itself barfed. but remember it was bobcat parser. todo: try with spider parser, spider ansatz
   	- update: we finally have an end to end - system.  working for spanish embeddings. dev accuracy with OOV model is 59percent
3. also, get gpt or fasttext embedding for english- and rerun food-IT again with OOV issue.
 	- megh asked: why not start with Glove. since its english- fasttext is not really having any semantic richness. So its important to keep it at word level. since QNLP is a model which relies so much on semantic/word level stuff- so maybe using fasttext (and even gpt) is a bad idea with QNLP
  	- nevertheless all these experiments should be done one way or other   
  	- maybe fasttext is useful for spanish, uspantek
   		- also another brilliant idea from megh is: maybe for new languages- it makes sense to use Byte pair based encodings.
     		- and/or (again from megh ) is, say you are given an entirely new language, and you want to know what the noun is...new language in the sense you dont speak or understand it. Say spanish. Bobcatparser combined with spanish tokenizer can still tell you what the noun is, what the transitive bverb is etc. This will be really useful if this can be extended to USPANTEK- because that means, a human curator doesnt need to necessarily understand or speak uspantek- or even a ML model, it will still give good results. Now note that all this is discussion level ideas- IFFF this is substantiated by results. For example if we can show that when using GLOVE+ bobcatparser we get higher accuracy that when using FASTTEXT or GPT based embedding+ bobcatparser, then we can argue that since QNLP/bobcatparser by definition is living in the semantic world (ofcourse with abilities to understand syntax also), then it makes sense to use a GLOVE kind of embedding which lives at semantic level (e.g king is closer to queen) as opposed to a n-gram based mechanism like FASTTEXt. Then we had the discussion on, but word level doesnt mean sentence level right. For example the sentence king is the ruler of a country and queen is the ruler of a country- inherently tells a human reader atleast, that king and queen are nouns. So shouldnt we want an embedding form that also understands it at SENTENCE level- so another idea is , can QNLP be used to create embeddings. - which can inturnbe used by NN based models. the advantage is, QNLP modesl function at the sentence level, and innately uses syntax, because of which in the aforementioned 2 examples, it goes closer to human intuition, tha tofcourse now it makes sense that King and Queen ka embeddings ended up next to each other. ..this is a very good discussion for future...
       - # todo
       - start with GLOVE on FOOD-IT and climb up to FASTtext and GPT- if nothing else/no ground breaking results, that itself will be a good paper/discussion.
       	- also do note that FOOD-It is completely living in classical land, i.e spider ansatz and pytorch models- so we might want to experiment Glove+FOODIT on quantum tariner, quantum models and IQP ansatz before calling it a failure/moving to GPT/Fasttext land
       	- find if in the FOOD_IT paper do they explicitly say that they know they are feeding val data during training itself. Either way, do drop an email or a pull request for their lambeq documentation - to explicitly state this fact. Otherwise this is pure cheating, where youare telling the world here is a pathbreaking model which gives 100% accuracy on 100 sentences, and you will still get another Mithun kinda poor guy down a rabbit hole for 3 years, because they blindly belived it was the models ability which gave 100% as opposed to the simple fact that they were showing val data to model during training itself.

## Nov 12th
### Mithun's logs
- We finally have one end to end system working for OOV. for FOODIT  using bobCatParser,SpiderAnsatz,PytorchModel,PytorchTrainer. Not that the accuracy on val was only 45%, but since we are still in sanity check land, am not going to tune/investigate that FOR NOW
- Next goal in sanity check: try to get values of Khatri back. Note khatri uses Depccg parser, IQPansatz, Numpymodel, quantum trainer. We definitely cant use DEpccg parser. Thats a nasty rabbit hole I dont want to go in.
1. change data to MRPC -original, not the hacked version we have been using for classification toy experiments
2. add in his change of quantum circuits including the equality_comparator
3. rewrite code for a pair of input, instead of just one
4. and try to use all models and parser same as his.

## Nov 11th
### Mithuns logs
- Found what is causing the issue in .eval(). Pytorch model has staggered entries i.e for each word the tensor length is different.
- However when you use OOV model, it flat predicts only 2 values. So if you hit a word with 4 tensor length, the 2 value is not enough to represent weights
 - Three solutions:
	- Easy way
 		- max params must be a value of product of dimensions of basic type, and the length of the dimension of the word i.e if bakes_2_n@n.r@s- and we assign n=dim(2) s=dim(2), we need to have a vector of size 4 prepared to store its weights
   		- so create dict2/np.zeroes based on that max value
   		- find why last layer of NN model is predicting 2 instead of 4 values (most likely linked to max params)
   		- or trained_qnlp_model itself has staggered params so why the fuck would your weight vector have only 2
	- Right way
		- try tensoransatz instead of spideranstaz. I Have a bad feeling spideransatz is not writing the params per word correctly. I dont know what spideransatz is or what spideransatz does, it was a vestigial choice from almost a year ago- because spider parser was the only one that was not bombing for the data we were using then. now bobcatparser is easily reading the data.-and our experiments arein classical land, so use the flagship of classical functors, i.e TensorAnsatz, (with bobcatparser, pytorchmodel and ppytorch trainer)- plus eventually once we move to quantum world, I think most of these issues will go. But even then its important that our foundation in classical equivalent (i.e tensors) is very strong.
  
## Nov 8th 2024
### Mithun's coding log
- Todo: 
	- add their differences 1,2,5,6 (from yesterday's notes) to our code --done
	- does it replicate an accuracy of one , even though the model is initlized with train, dev and test circuits. i.e dont test on val during training but instead let it run for 30 epochs,and use that model to instead test on val- Ans: yes, gives 100% accuracy
- English
	- add our OOV trick and see what accuracy you get for food IT (without all circuits initialized inside model 1)
		- this is to check if person_0_n@n.l present in val issue will be taken care of by our embedding
   		- update: getting this error:  `shape [2, 2] is invalid for input of size 2` inside trainer.fit()
			- sounds like label vs dimension issue. update: found out what the issue is.
     			- The dimension for each of the entry in qnlp.weights, must exactly match that of initial param vector. Infact that dimension is  decided by:
				a. how many qbits/dimensions you assigned to
                    n and s during the ansatz creationg an
          			b. how complex the word's representation is. for example let's say you gave n=2 and s=4 qbits. So john_n will have a dimension of 2, since it has only one noun. however, now look at prepares_0_n.r@s. This will have a dimension of 8 because it is the product of a nount and a sentence. therefore 2x4=8. Therefore the initial param vector also should have a tensor of dimension 8. i.e in the below code, am hard coding exactly 2 dimensions for all words. THAT IS WRONG. the number of dimensions must be picked from qnlp.weights  and the a initial parameter prepared, by that size. Now note that khatri is simply picking the  first n cells of the embedding vector- it is as good as initializing randomly. that's ok he has to start somewhere and this is a good experiment to mix and match. However, first and foremost the dimensions hs to match
		- manually look at the corresponding weights and pass this bug. this is a band aid. Ideally, todo: understand deeply dimensions and qbit assignments
         	- update: getting another error at model3 (nn).fit- expected shape[2] given shape[3]
          	- solution: this was because i was creating the vectors in NN_train_Y using maxparamlength+1- should have been just maxparamlength alone
          	- update: passed nn.fit() error for model3. Now another error inside evaluate_val_set. `index 2 is out of bounds for axis 0 with size 2`- Solution: this was because i was hardcoding parameters to be always a tuple of 2. that is not true- it depends on the _0 and _1 ka value.
          	- update: error in pred_model. get_diagram_output(val_circuits) which is `RuntimeError: shape '[2, 2]' is invalid for input of size `
	-  inject oov words in val to their data and try above
 	-  why do they pick epoch 30- add early stopping or atleast plot dev and train accuracies/losses and pick a decent epoch yourself.
  	-  reading
  		- todo of 1: read more on sentences2diagram
  	 	- todo of 2: why ansatz dimension 2 and 2
  	  	- todo of 5 above: why no remove cups
  	   	- todo of 7 above: why staggered weights  	
 -  Spanish
 	- todo: using Changes  1,2 and 5bobcatparser- run our code with spanish data.
  
## Nov 7
- Discussion on Fasttext embeddings
	- For spanish, we have an executable `.bin` file. It can't be opened, but when we execute it and provide it a word, it will return an embedding.
 	- English has word-level embeddings, Spanish has n-gram level embeddings
- details from Mithuns coding
  
	- able to replicate the Food IT classification using our code.- the file is called: replicating_food_it.py
 	  - however few things to note
 	  - Qn) Does their code barf if we provide a new unknown word in dev or tst?
	  	- ans: No
 	  - Qn) Why?
 		- Ans: because they are "smartly" using circuits of val and test data during initialization of the model
 		- i.e `all_circuits = train_circuits + val_circuits + test_circuits`
 		- `model = PytorchModel.from_diagrams(all_circuits)`
		- Qn) will their model barf/complain about OOV if we initialize it only on train_circuits.
			- Ans: yes
		- Qn) does Khatri initialize it with only train_circuits or all?
			- ans: only train_circuits
		- Qn) what accuracy will we get on val data, if we initialize only on train_circuits, and do the training for 30 epochs and then use that model to test on val?
			- ans: hits lots of OOV, but this time at an interesting level. person_0_n was a symbol present in the trained qnlp_model.symbols and had a weight corresponding inside qnlp_model.weights. However, the word person in val was n@n.l...or something complex. and that was called as OOV...fucking ma ka lavda. their code is dumb as fuck.
		- Qn) same above scenario what accuracy do you get at the end of 30 epochs on training data?
			- ans: 92.86 percentage (obviously, overfitting)
		- Qn) what other major changes/differences are there between their code and ours
			- ans:
				1. they use sentences2diagrams while we use sentence2diagram. todo: Use their method. Our way was giving arrow/cod/dom level errors.
				2. in ansatz definition they give dim 2 for both noun and sentence. we were giving 4. If I gave 4 in their code, error occurs. weird. todo: read more on this. Thought sentences were supposed to live in a dimension higher than nouns as per lambek
				3. they initialize their model on all_circuits like shown above. TODO: Nothing in our code/continue initializing only on training
				4. they pass val_dataset during .fit() function itself. TODO: Nothing. Our code will barf due to OOV. SO its better we keep training and evaluation separate
				5. they dont use removecups -atleast not in classical todo: find why
				6. They use bobcatparser instead of spiderparser. Everything else (spideransatz, pythontrainer, pytorch model remains same as ours)
				7. their qnlp_model.weights have staggered sizes. i.e some words have tensor of 2 while some others have tensor of 4. i think this is dependant on how a word is finally converted using lambek calculus. i.e if there is just one basic type n or s it will use 2 dimensions (since everything in foodIT code is in tensor level) while complex ones get more.example below. Todo: read and understand and debug more into this.
for example:

```
qnlp_model.symbols[24]
woman_0__n@n.l
qnlp_model.weights[24]
Parameter containing:
tensor([ 0.8845, -0.4794, -1.3659,  1.3689], requires_grad=True)
qnlp_model.symbols[23]
woman_0__n
qnlp_model.weights[23]
Parameter containing:
tensor([-0.4763, -1.8438], requires_grad=True)
```

## Nov 5
- ToDo Megh:
	- set up ML flow for the project
	- Start working on the NAACL draft
 	- Work on a for-loop of model requirements
  	- Find english embeddings for testing the english model
  - Tested `requirements.txt` setup in a new environment
  	- Code uses cached `numpy`, and fails to install spaCy.
   	- `matplotlib` issues
    	- Solution: created a shell script to install all requirements in python 3.11.10
     	- Merged all changes into `main`
    - ToDo English embeddings: find them online, and if not, figure out how to incorporate english vector files into our code.
    - Chronology: word2vec, BERT, Fasttext, Byte-Pair (used by GPT). We would ultimately need the n-gram embeddings. Fasttext is used to build words, by training on natural language, n-grams, and thus creating relations between words. Richer because it has seen more data
    	- Why do we need gpt embeddings- they have learnt word meanings after a level of training on the sub-word embeddings.  

## Nov 4
- Model breakdown:
	- Model 1: trains on circuits (QNLP model)
 	- Model 2: fasttext model for OOV tokens
  	- Model 3: simple feedforward model with early stopping
  	- Model 4: runs the actual classification task
- October 31st deadline met. Now need to continue with hyperparameter tuning
- English data results
	- MRPC corpus: maps a large chunk of text to a summary. Needs code modification
 - English data:  

## Oct 30th 2024
- got code to train on OOV model also. First time ever. Now trying to use it to predict on val set.

## Oct 28th 2024
khatri code is also breaking at .fit() with the error `raise ValueError('Provided arrays must be of equal shape. Got '
ValueError: Provided arrays must be of equal shape. Got arrays of shape (30, 2, 2) and (30, 2).`

update; fixed using the same issue below from last week-i.e using BCE loss

update: khatri code is breaking still at first .fit saying;
```
File "/Users/mithun/miniconda3/envs/qnlp/lib/python3.12/site-packages/torch/nn/modules/loss.py", line 731, in forward
    return F.binary_cross_entropy_with_logits(input, target,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/mithun/miniconda3/envs/qnlp/lib/python3.12/site-packages/torch/nn/functional.py", line 3223, in binary_cross_entropy_with_logits
    if not (target.size() == input.size()):
            ^^^^^^^^^^^^^
TypeError: 'int' object is not callable`
```
switching to our v7 code: still getting assertion error for not circuit as input. WTF?

## Oct 27 2024 
Mithun self hacking
Went through khatri code debug line by line 

In khatri code, his vocab has `friday_s` while our code is removing the `_s` part.

`train_vocab = {symb.name.rsplit('_', 1)[0] for d in train_circuits for symb in d.free_symbols}`

Also `max_word_param_length`- make sure spider ansatz gives some number- maybe try it out in ipython, or switch to tensoransatz.
If we are using Quantum trainer, why exactly are we doing classical parsers? why not switch to bobcat?

 `max_word_param_length = max(max(int(symb.name.rsplit('_', 1)[1]) for d in train_circuits for symb in d.free_symbols),`
                            `max(int(symb.name.rsplit('_', 1)[1]) for d in test_circuits for symb in d.free_symbols)) + 1`

Qn) what is the exact status of our v7 code as of now
ans: assertion error inside first fit? thought we moved beyond that long time ago
  
## Oct 25 2024
- Continuing discussion from last time on the basics of lambeq's formulation of language.

## Oct 21 2024
- Response to Gus Han Powell:
	- Why is this called "quantum" NLP when we don't use quantum computing?
 		- Theory of quantum physics is key: qbits, angles
   		- Lambeq calculus is the mathematical framework, developed along with category
     		- Complexity of the problem determines whether we use an actual QC or a classical computer. As our sentences become more complex, we need more qbits, thus an actual QC
       		- LLMs look at probability of a word with its neighbor (a counting machine, bottomup approach), this is not used for predictions in a QNLP context. Can an LLM differenciate between a relative subject vs relative object?   
	- Why do we need a Lambeq framework?
 		- QNLP needs category theory 
 		- We do not use Chomskian grammar as it is not compatible with Quantum Computing (because QC needs category theory, which uses sets, groups and functions, dependency frameworks does not map to category theory)
   		- HCG parsers with long sentences cannot be accomodated with hardware
         	- Lambeq framework: a sentence can contain items from three categories, S, N (anything identifiable as a noun), N/S. It also needs rules to solve chunks to obtain these categories. A given chunk is solved according to the rules in the framework.
     		- The sentences are parsed to S and N: every S is broken down to S/N, until no S are left
       		- Thus, simplifying sentence parsing is possible 
	- What libraries and code are we using?
		- We use the python library lambeq for this 
     		- If we move to QC fully, we would use the lambeq library's BobCat parser, which utilizes this framework: it takes the output of a parser (designed based on Lambeq's N/S framework) and turns them into quantum circuits, which can be used downstream
       			- Converts sentences to diagrams, and uses a ZX diagrams and simplifies sentence diagrams to reduce no. of qbits needed. 
       		- While using classical computing, we use lambeq python library's spider ansatz: Very simple framework that connects each word in a sentence, ranked equal, to one label
         		- It reduces the number of qbits needed, and is linear and grows linearly. Just a test case
           		- Spider Ansatz works like bag of words, so it can be run with an LLM. It's a classical baseline that we use to test our system and have a baseline to test a quantum computing setup 
		- We still use PyTorch tensors, and code in python.
 	-  What is the "theory" that gives us great results?
  		- Classical system has only 0s and 1s, so no. of parameters needed to encode and process NLP parses and relationships is in the billions
 		- But qbits are a good optimized way to do this, hence lwss data needed for training, and less overhead. As sentence grows, no. of needed qbits grows linearly, rather than exponentionally, as with embeddings
   	- Why do we use a classical computing system, instead of a quantum computing?
   		- IQPAnsatz in the lambeq package can give us a complete QC system
   	 	- Current classical system can process and output the quantum circuits into tensors, it is a vector
	- How do we know that what we are running is a QNLP ?
 		- IBM quantum cloud- runs for free, also provides a simulator
  	- What is our research problem?
  		- QNLP uses small datasets, so out of vocabulary problem neds to be resolved
  	 	- Solve this for other languages
  		- Actually understanding the results   	  

## Oct 18th 2024
## Mithun's self hackathon
- in mithun_dev branch
- Our code (v7_merging_best_of_both_v6_andv4) is still stuck at the dimension mismatch between model.weights, and the parameters he is passing
- so today to find the expected value, i thought i will debug  khatri's original code. (it is now kept in archive/original_code_by_khatri_oov_mrpc_paraphrase_task.py'.-
	- **update**: got it to work past initial .fit() of model1 and even started training of DNN model. This is the first time in 3 years...phew..hai esperanza...either his code, or our code one of them will cross till the finish line soon. note that this is using the datasets `"data/mrpc_train_small.txt` and `"data/mrpc_dev_small.txt"`. That after bobcat parser throwing things out has just 1 and 1 in training and test. meanwhile if we use `"data/mrpc_train.txt"` `"data/mrpc_dev.txt"`- getting broadcast error. weird.
 	- update: both `*_small` will give good positive count for train and dev if we use `MAXLEN = 20`.
  	- update: still getting the input mismatch in loss
    	- update: oh that's because we are using BCELoss from torch. switching back to his custom loss
		- update: getting the old error of `loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss ValueError: operands could not be broadcast together with shapes (30,2) (30,2,2) ` note that the 30 here is batch size. so somewhere the dimension between gold y and preds y_hat are getting mismatched..update: this is the issue with how khatri is writing the loss funciton. Just use bce from torch instead.
  	  - update: that fixed the issue. i.e using our own accuracy instead of what khatri defined. Now getting another error
  	  -
  	    ``` File "/Users/mithun/miniconda3/envs/qnlp/lib/python3.12/site-packages/torch/nn/modules/loss.py", line 731, in forward
    		return F.binary_cross_entropy_with_logits(input, target,
           	^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  		File "/Users/mithun/miniconda3/envs/qnlp/lib/python3.12/site-packages/torch/nn/functional.py", line 3223, in 			binary_cross_entropy_with_logits
    		if not (target.size() == input.size()):
            	^^^^^^^^^^^^^
		TypeError: 'int' object is not callable`
  	    ```	
	- Note that all this was fixed in our v7 code, just that we neve documented it.
  	  - i think its time to read the LAMBEQ documentation again. Had last read it in 2022, when my knowledge/consciousness was much much less. So I think its time to read it all paperback to paperback again.
- lots of good findings
	- his train_vocab is written as `'meanwhile__s'`
	- he is literally taking the idx value of the fasttext embedding in this code. weird 	`initial_param_vector.append(train_vocab_embeddings[wrd][int(idx)])` i.e initial_param_vectoris just a list of floats
	- qnlp_model.weights is also a list of numbers...WTF
	- `[0.5488135039273248, 0.7151893663724195, ..]`
	- Also len(qnlp_model.weights) =48=len(initial_param_vector)
	- so why is our qnlp_model giving a list of tuples instead?
	- 48 is the total number of symbols, which i think in our case is 408 or 463
	- its easy if we know what the qnlp_model does when it does .fit(). But that is bottom up reading, which is at 95%
	- meanwhile we are approaching it brute force/top down/nasty analytical skills only based coding.
	- next find if qnlp_model using pytorch is different in weights than when using numpy model
		- answer: at initialization both are empty lists
 		- getting error with numpy model when used with pytorch trainer. its expecting same trainer/compatible trainer like quantum trainer
  		- why not use numpy model and quantum trainer in our class, and some cross .fit()- we just want to see if the weights is any different. answer: yes that fixed the shape[2] error. But now getting assert circuit error. update. Shape[2] error almost always means the weights of the qnlp.model is expecting a tuple of 2 and you are giving it just 1. or vice versa
  	- update getting the below error in first trainer.fit()
  	- ``` line 140, in get_diagram_output assert isinstance(d, Circuit)
		AssertionError
  	  ```
	- i.e something in the train_dataset is not a good circuit..
if nothing else this we can do dataset by dataset comparison and find out. what is different between khatri code and our code.
- todo:
	-  debug both code line by line and find if everything matches, including the idx, _s and even the size of weights and params
		- his max param length is 2+1 =3 because there were values like `Thursday__s_2')`. Note that i am even using spider reader. so why is our spider reader not producing _2. i.e parsing different words to same time
		- his wrd= `Friday__s` while ours is `Friday`
   
## October 16th 2024
### HACKATHON
- Originally, Mithun fixed the `.fit()` by simply not inputting any embeddings, and only working with the text in the dataset
	- temporary check 
	- Is the issue with `numpy` array size? No, `lambeq` is having the problem
- Running no embeddings model to try permutations and combinations and assess performance
	- [Commit with working code](https://github.com/ua-datalab/QNLP/commit/c4e56a1746965c4c12876fbf67c4fb0ff463a845)
 	- training working, testing did not 
 	- Confirming the OOV issue: added all data from val set into train, no OOV issue. Model is able to memorize
 	- `nn.loss` bug: numpy doesn't like torch tensor. Switched to `torch.nn.BCEWithLogitsLoss()` instead of custom loss function, so everything is in the same format
 	- validation list was not a tensor, but a python list. Wrote a wrapper to fix this: `val_labels_pytorch_tensor= torch.Tensor(val_labels)`
- Getting the model to print accuracy and run `model.eval`.
  	- Wrote a wrapper that can calculate the accuracy and outputs a dictionary with metrics
   	- All sources of bugs during weight assignment ruled out. Only problem is passing OOV words in the format the code needs
    	- Train vocabulary: to-do. Regex does not remove punctuation and parenthesis in `wrd`. Fix it
     	- `qnlp_model.weights` is an array. It is asking for a tensor that requires grad.
     	- However `initian_param_vector` that is passing values to it is a vector.
     	- So we are facing a mismatch. when trying to append the latter to the former.
  	- Check if `self.symbols` and `self.weights` have the same dimensions
- lambeq's code needs a dictionary with word and its weights. Which is an issue for OOV words the way the code is written 
-  `val_embeddings` are actually not being called anywhere- we need to pass this to the model
-  FIX: For QNLP model, DO NOT EVALUATE USING THIS THE QNLP MODEL. This is not the model designed by khatri et al, so it is partially trained and not set up to handle OOV words. Khatri et al. predicts using a "SMART" `OOV_prediction` model.
-  BOTTOM LINE: the training QNLP model gives a 72% accuracy on the test set. It learns, but isn't doing a perfect memorization. We need to complete the pipeline to see performance on the dev set. 
 
## October 14th 2024
- Mithun canceled meeting due to family emergency (dog not well).
	- However Mithun did start the experiments. Results are kept [here](https://docs.google.com/spreadsheets/d/1NBINiUsAdrqoO50y_CX_BGGgXcP9Zt6i5nYKvuB70Tg/edit?usp=sharing) inside the tab titled "oct14th2024_noEmb"
- Will be using the version of the code kept [here](https://github.com/ua-datalab/QNLP/blob/mithun_dev/v7_merging_best_of_both_v6_andv4)

## October 9 2024
- Background: When megh and mithun met on wed oct 9th 2024- we had two paths we could take.
	a. there was an investor ready to jump in if we could show that QNLP is great off the shelf for native american languages.
		- Earlier results were conducive. Right now the status of the code is that, it works end to end without embeddings or khatri's 4 model solution.
  		- However, we were thinking of give a week of status quo/experiment/parameter search/fine tuning to ensure that out of the box (i.e only 1 off the shelf model in khatri code) works.
  		- Goal is between october 14th and 18th, we take the code and run it till plain QNLP model train+dev- spread it across various ansatz and diagram convertor, try it on a)uspantek b) english c) spanish, and find the max dev accuracy.
  		- If nothing inteesting shows up (i.e no high accuracyies/above 80%) we will continue with
  	b. incorporating embedding path, Planning for next week
- plan for week of oct 14th to 18th 2024
- Monday: hackathon for analysis and reporting the performance of the no-embeddings model.
	- Access F1 scores and analysis of the confusion matrix
 	- Visualization and a descriptive write-up of the results
  	- Look at performance of >1 ansatz
  	- Try the code on the test set
  	- Make a plan for processing some of the other documents on different topics

## October 7 2024
- Production side of QNLP project
- Prepare production materials and project analysis
- October 31st deadline for a QNLP product, in order to demonstrate capabilities
- Status of code
	- Working setup for English- used the GPT model
 	- spanish and english models without embeddings works, but poor performance
  	- embeddings model stops at `model.fit()`
- Options:
	- Try functioning models with actual quantum hardware.
 		- Strong chances for English or Spanish

## October 2nd 2024
- Discussion of category theory
	- Concepts: magma and "pregroups" (set theory)
- Bug while runing `fit()` on model 1: tensor shape does not match expected shape for `train_embeddings`
	- weight assignment?
  	- OOV issue with special symbols?
	- `qnlp_model.weights`: initial parameter vector= Fastext embeddings. This is a list. But the shape is not matching with expected weights. Issue with pytorch- it is an executable, so cannot open it in debug mode and assess what shape is required.
 		- since weight assignment (for QNLP model?) is causing a bug
 	- `train.datasets`- has 85 sentences with <n words. Length is same as `train_labels`.
  	- `train_labels`- requires a matrix. Label 0 is [0.0, 1.0] and Label 1 is [1.0,0.0]. Mithun was able to provide the same format.
  	- Khatri et al. uses an equation to define loss, we replaced it with `torch.nn.loss()`
- ToDo: what is shape `[2]`? Where is it being defined?

## September 30th 2024
- Debug trainer"
	- `sym` explantion- `aldea_0__s`- word, label, ways in which it can be expressed in lambeq
	- The embeddings from Fasttext model (2nd model initialized) are used as initial parameters of the qnlp model
		- Get QNLP model initialized with embedding weights
	 	- Trying to assess the issue with assigned weights 
	- initial parameter shape mismatch, shape of weights array does not match requirement
 	- Model 1 `.fit()`:
 		- if weights are updated correctly, then there should be no OOV words in train 
 	- Current objective- why is the validation code buggy? Khatri et al. is not evaluating at every epoch, but Mithun was trying to do that
  		- val set has a lot of OOV words, so if it is run along with training set, we will end up with a lot of OOV words and bugs
    	- solution- don't call val dataset, as it has OOV words. Call it later. Optional parameter for `fit()` so no issues
- What does khatri et al. mean by "weights"?
	- ToDo- Mithun  
-  Readings
   	- Todo: 1999 lambeq grammar, 2010 (start here)    

## Sep 26th 2024
### (Mithun explaining the work flow of khatri's code in a question answer.) How Khatri et al.,works
- (especially the original [code](https://github.com/ua-datalab/QNLP/blob/mithun_dev/archive/master_khetri_thesis%20.py) 
- Basics:
- ANy neural network/machine learning model does same thing; i.e given two things a and b, find any patterns that relates a to b. For example if the task is given a ton of emails marked by a human as spam and not spam, train from it so that when the model sees a new email its job is to predict whether it is belonging to spam or not spam. However, during training the model is provided two things, like i just mentioned a, b i.e model(a,b). In this case a will be an email from teh training set, and b will be the corresponding label (spam or not spam) which teh human had decided. Now the job of the model is to find two things a)what is it/what pattern is there in the data that makes this particular email be classified into class spam (for example) b) what is the common patterns i can find inside all the emails which were marked as spam.At the end of the training process, this `learning` is usually represented as a huge matrix, called weight vectors or parameters, which if you really want to know are the outgoing weights that a neuron assigns to the outgoing connection between itself and its neighbors.
- Now with that knowledge lets get into the details of this project-"how this code runs"
- There are 4 models that are being used in this code
	- Model 1 QNLP Model
 		- First one is the the main QNLP model whose job is to find a relationship between input sentence and the label, for example 	class A (exactly same as a neural network model). For example one of the data we use are 100 sentences in spanish- which are classified into two classes , dancing and education.
	 	- so the job of QNLP is learn during training model(a,b) , where a is a sentence in spanish, and b is the corresponding class label (e.g.Dancing). This is being done in [line 503]([url](https://github.com/ua-datalab/QNLP/blob/mithun_dev/archive/master_khetri_thesis%20.py#L503)). Note that by the time code reaches line 503 all training is done as far as QNLP model (the first model) is concerned, and it has learned what it takes for a sentence in spanish to belong to the class education (or dancing). (but no prediction or testing is done on the test partition), Or in other words, once the learning/training is complete, just like NN case mentioned above, the system produces a huge matrix called weight matrix to represented what it has learned- the pattern which makes a given spanish sentence to belong to the education classy, say...very similar to the same `what makes an email belong to class spam`. Only difference here is this weight matrix is not really the weight of the neuron in QNLP instead these are called angles of the rotational gates. That is because instead of neurons QNLP model uses internally something called quantum gates.
 		- Note that only difference between QNLP model and a standard neural network is that standard neural network expects input in terms of vectors filled with numbers, while QNLP model expects input in terms of something called circuits. You dont have to worry about it for now.  We use something called [ansatz]([url](https://github.com/ua-datalab/QNLP/blob/mithun_dev/archive/master_khetri_thesis%20.py#L475))- (which is yet another tool from Quantum physics, which you can ignore as a black box for now) , which internally converts a given sentence to a circuit
  - Qn) great but that is model 1,..what was the other 3 models, and why were they needed? things seem to be almost done so far as far as traininng is concerned right?
  - Ans: very good question. Except, when the author used the aformementioned trained model to predict on the test dataset, he ran into a very weird problem. There were some words in teh testing set, which were not present in the training set. Which means there is no corresponding mapping between this word and an entry in learned matrix aka the weights aka the angles of gates. (Todo for mithun- confirm this. I still dont understand how words can have weights).
  - Anyway, this is called the out of vocabulary problem, which is a solved problem in classical NLP. Rather it took almost a decade to solve this problem. When in 2013 initially this problem occured, people ignored it, mainly because a) it was a very rare phenomenon. Rather the training corpus was so huge that it was very rare that a word wil be encountered in the test set which was not present inthe training set. b) the way people ignored it was by creating a tag/word called UNK (stands for unknown) and assigning all new OOV words that label and its corresponding weights.
  - as you can see that was a disaster waiting to happen. Not all UNK words mean the same. This became a big problem in a low resource seeting (similar to teh 100 spanish examples above ) because when the training data is very small, there is a high probaility that the word in test set is never seen before/ OOV
  - That is when the whole concept called Embeddings were invented. The idea of embeddings is simple. Take every single word in the dictionary and pre-create/beforehand itself create a corresponding vector for it. This is typically done using techniques called CBOW (continuous bag of words) or Skipgrams if youw ant to know exact details.
  - However even that hit a huge block especially in case of languages other than english. For example in spanish, even if the embedding model has seen most of the words in the languaage(e.g. el, para), the testing set might still have an entirely new rare word (e.g., pardo- a color between grey and brown). This was even happening in ENglish. So as a solyution to this problem someone invented a technique called fast text embeddings (which eventually inspired byte pair encoding, which is used in Transformers/LLMs). The intuition is that instead of learning embedding for every single word in say ENglish, they learn embeddings for n grams. For example instead of learning the embedding for a word `arachnophobia' the new model will instead learn embeddigns for `a',`ar',`ara' etc...i.e oen gram, two gram 3 gram etc. The advantage of this approach is that even if an entirely new word is encountered in the testing data, and even if the word was not seen before/part of the embedding model, it can still be built up by the n-grams.
  - Anyway long story short, we also encountered the OOV problem, and we decided to use FastTExt. I.e give every single word to fasttext, and get its corresponding embedding,. This can be seen in line [337](https://github.com/ua-datalab/QNLP/blob/mithun_dev/archive/master_khetri_thesis%20.py#L337) where the fasttext model is initialized and in line [130]([url](https://github.com/ua-datalab/QNLP/blob/mithun_dev/archive/master_khetri_thesis%20.py#L130)) where every single word in training and testing is converted to its corresponding embedding.
  - Qn) But i thought you said your QNLP model takes as input some circuits thingie and outputs angles of the gates in these circuits. What will i do if you give me a vector for a word.
  - Ans: Very good questions. This is where model 3 comes up.
  - the author (nikhil khatri) created a simple Neural network based a 3rd model whose only job is to find pattern between embeddings and the angles.
  - Qn) I dont get it. how does that help.
  - Ans; Remember we were saying that by the time the code control executes [line 503]([url](https://github.com/ua-datalab/QNLP/blob/mithun_dev/archive/master_khetri_thesis%20.py#L503)) only TRAINING part of QNLP model is done. Now say the same model is going to be used for prediction in the test set. What is the meaning of testing. i.e we give same input like in training (e.g. circuits corresponding to a new sentence in test set). Which will inturn be used to multiply with the angles of the gates of the learned model, (equivalent in NN world will be multiplying embedding of a test set ka word with the weights of the learned model), and get a float value, using which the Model decides if the test data point belongs to class A or B  
  -  and it encounters an OOV word. So it goes back and asks the 2nd model, the fast text embedding generated model, and asks- here is a new word, can you give me the corresponding angles for it. So the model 2, does exactly that and gives out a vector. So then model 1 asks- WTF am going to with vectors, i only know circuits as inputs. That is where Model 3 comes into picture. So to remind you model 3, is a model which tries to find patterns between model3(a,b) where a is embedding and b is the weight equivalent(mithun todo: Ideally it should have been finding pattern between Embedding and circuit.. am still not completely clear on why model 3 outputs patterns between embeddings and weights/ angles of QNLP model instead of circuits- go find and update here). ANyway what happens is, before model1 does any prediction, we train model 3 between two things, the embeddings coming from fast text for each word in training data set, and the corresponding angles which we get from the TRAINED QNLP model, which is model 1. Specifically, in [line 197 ]([url](https://github.com/ua-datalab/QNLP/blob/mithun_dev/archive/master_khetri_thesis%20.py#L197))and 198 is where we initialie the inputs a, b to model 3 are created, i.e., the embeddings from fast text and the weights from the QNLP model. Then training of this third model is trained in [line 218]([url](https://github.com/ua-datalab/QNLP/blob/mithun_dev/archive/master_khetri_thesis%20.py#L218))
  -  Now once the third model is done training
  -  Qn) ok then what does model 4 do
  -  Ans; Short answer it is yet another NN model which is purely used for prediction
  -  Now consider line [248]([url](https://github.com/ua-datalab/QNLP/blob/mithun_dev/archive/master_khetri_thesis%20.py#L248)) the first thing they do is, take every word in the test vocabulary, and gets the corresponding embedding of it from model 2 and then gives it to model 3 who returns with the correspondingn weight that model 1 understands.
  -   All these weights/angles which is taken out of model 3, is used to initiate model 4.
  -   This model 4 is something which takes test_circuits as inmput (just  like  QNLP model) and predicts output (which is which exactly model 1 does- however, now remember there is shit load of embeddings involved, which is hidden from you)  at happens in [line 264](https://github.com/ua-datalab/QNLP/blob/mithun_dev/archive/master_khetri_thesis%20.py#L264)
  -  Thats how the whole system works on 4 models.
## September 25th 2024
- Potential solution for OOV words in the Spanish model- why is the model not performing well with embeddings?
	- model is given the word's vector representation. in the case of OOV words,we are providing embeddings
	- Q: are these two types of input in the same format? A NN is multiplying given vector with weights to find correlations 	
- Unlike English, the load of OOV words is very high, so the model will need to find a solution and rely on embdeddings
- In QNLP context, we have circuits and a combination of angles. How do we convert embeddings into a combination of angles?
	- Khartri et al.'s solution- use a neural network model that finds patterns between embeddings and angles, so that this conversion is aided by model making predictions.
	-  Now, we need to test each model to see where the bottleneck is.

## September 23th 2024
- Created a flowchart explaining Khatri et al.'s code, to understand each process in the model used
	- Why is there is NN model (MLP) between the input and QNLP model?
		- Section 62- NN is learning a map from embeddings of vocalbulary in training set to QLP model's trained parameter assignment. This helps "generalise unseen vocalbulary".
	 - Does the Uspantekan model without embeddings require an NN at all? No, because there are no embeddings to map.  
	 - What are the input and outputs to the QNLP model with embeddings?
  - Steps for training:
  - For every word in training vocabulary:
  	1. Thread 1:
   		1. Get embeddings from Fastest
   		1. Generate vector representations
   		1. This is a **pre-trained model** 
	1. Thread 2:
  		1. Initialize some non-zero weights for the QNLP model
   		1. In parallel, run the training dataset through lambeq ansatz, and obtain circuits at the sentence level.
   		1. Use these circuits as training data and **train the model** to get weights
   		1. Obtain word-level representations from the model weights. TODO: how do we generate word-levels representations from sentence-level trained weights? 
	1. Bringing it together:
   		1. Create a dictionary by mapping each word in the vocabulary with weights from the QNLP model (which is also a vector representation). DON'T KNOW HOW
   		1. Create list of lists mapping each word to a vector representation of its fasstext ebedding. TODO: check how this is done.
   		1. **Train a model** to find a relationship between word-level vector representation from QNLP model weights, and the embeddings from fasttext
   		1. **Train another model**: TODO- find out why
        1. Testing the final model
        2. 
## September 18th 2024
- Discussion of code v7
	- `train.preds()` did not work: Pytorch tensors include the loss values plus the computational graphs, so there would be an issue running them with numpy
 	- Solution: `torch.detach()` which creates a copy of the tensor, without the graph
  - Substitute hard-coded loss function with `torch.nn.loss`
  - Fixed error that combined train, test and dev for generating training data
  - what is `embedding_model`- fasttext
  - ToDo: draw a flow diagram to understand what `qnlp_model` is doing
  - DNN model: training between word and its embedding
  - QNLP model: angle and word
  - ToDo: find difference between symbol and word
  - ToDo Mithun: refactor code to ensure all functions are in the correct order, based on Khatri et al.

## September 16th 2024
- Tech Launch Arizona funding
	- AI + Quantum applications
- QNLP applications in Megh's dissertation
	- LLMs are trained on next sentence prediction, entrainment looks for people sharing linguistic features
- Discussion on QNLP applications:
	- RAG also a good fit for indegenous languages? May be a low-lying fruit, even if it runs into data issues
- Sprint: debugging v7 code
	- What does maxparams do? Spanish Fasttext model provides 300 parameters. So, the number is hard-coded into the code
 	- OOV words were an issue, changed max_params to fix this
  	- Setting up loss code and accuracy code
- Pytorch vs Keras
	- Using pytorch for calculating angles in place of numpy
 	- But if keras is being used   

## September 11th 2024
- Code sprint and logging and documenting bugs
	- Spanish embeddings model:
 		- added v7 to `megh_dev` and adding documentation.
   		- See: [code compare for more](https://github.com/ua-datalab/QNLP/commit/3472a3addd948076677e83c70dfd36892b38c41a) 
		- 1 AI model + MLP for mapping spanish embeddings with spanish text
  		- Initial model fixed by switching from numpy to pytorch

## September 9th 2024
- Uspantek + spanish
	- run more experiments to confirm the results 
   	- cons: Mithun wants to read more before turning knobs.
- write paper:
	- reproduce results again?
 	- confirm with robert, 
 		- we can use the uspantek data.
 		- remind him to connect with his collaborator's phd student.
- upcoming deadlines
	- NLP
		- COLING: sep 17th 2024  
		- ICLR: 2nd oct 2024
		- NAACL: Oct 16th 2024.
	- Computational linguistics
		- SCIL- december 2024
		- [LSA](https://web.cvent.com/event/40d9411e-b965-4659-b9c3-63046eeed3d4/summary)
- update: we decided to start writing paper- kept [here](https://www.overleaf.com/4483532232tcfnfdrrcbdc#12a1b4). AIm is for ICLR 2nd october. But more importantly, it is an exercise to capture all the thoughts fresh in our head
- Paper to-dos
	- Write introduction
	- get a lit review for:
 		- QNLP,
   		- Uspanteko,
		- low-resource language LLM research:
  			- https://arxiv.org/pdf/2406.18895
     		- https://aclanthology.org/2024.americasnlp-1.24.pdf
        	- https://arxiv.org/pdf/2404.18286  
  	- Results and methods
  		- Baseline models for Spanish, Uspantekan
  		-  Future: Fasttext embeddings for Spanish model
- Current issues:
	- Embeddings model has a lot of bugs
 - todo
 	- wed meeting,
  		- Mithun will try to push through code of spanish + embeddings, with the aim of: lets have closure/have 3 systems working
  		- Mithun: find more AI related NSF - QNLP for indigenous languages
   	- 	 

## September 5th 2024: Meeting with Robert
- Dataset has other topics related to education and dance: like "teaching Uspantekan", and other forms of dancing
	- We have more data! 
- [Link to Slides](https://docs.google.com/presentation/d/1jw9_b55BC4HOMmtOaqbXjDkOR8nn9_Zg8xLjK86KG8w/edit)
- Robert's update:
	- Currently: 1800 sentences, 12k tokens, with dependency parsing
	- 10k tokens for different discourse types
	- Plus 5 other languages with spanish dictionary  
	- QNLP has its own parser- but throws out a lot of sentences which it can't parse
	- Super helpful update about pre-parsed sentences!
- Target: which NSF project should we consider?
      - "Dynamic language infrastructure and Language Documentation" grant currently funds Robert, along with CISA (proposed by NSF).
	- Maybe a good option
	- Target date: Feb 18th 2025 [link](https://new.nsf.gov/funding/opportunities/nsf-dynamic-language-infrastructure-neh)
	- Better than AI grants, dominated by LLM
	- NSF doesn't like to fund the same grant twice- so keep both projects meaningfully different!
		- Check if QNLP can be imagined as an extension or addition to an NSF grant
		- Extension with more money- check if it's a thing for linguistics-focussed NSF, Robert could also get more funding. 
- Moving away from dataset creation grant, to a new theoretical framework (QNLP)
	- Work on more languages, so focus is on low-resource languages generally
 	- Another important focus: focussing on why this technology is a good fit for a particular use-case (low resource languages), and target tasks that human annotators are really struggling with
- Ex. automating: morphological analysis, parsing, POS taggging, spell-checkers, word search 
- What is the ask in terms for funding?
	- Personel: funding for Mithun, an RA/postdoc
	- Robert's experience: his share is 135k, most of it goes to the collaborator's funding, plus annotators
- Other next steps: check out of the box QNLP can help with tagging
   - Compare how the current AI automatic tagger trained by Robert's collaborator does compared to this
   - Mithun: read Robert's grant

## September 4th 2024
- v4 code running end-to-end. Why?
	- Code was stuck before `fit()`.
 	- Code switched from `numpy` to `pytorch`, needed to switch from arrays to tensors
  	- Ansatz- issues
  	- Parser language settings- when the parser encodes
  		- Embeddings from fasttext was different from what `lambeq` expected:
  	 		- Parser from `lambeq` adds underscores and to each entry, which is missing in the fasttext embeddings. Needed cleanup to prevent mismatch.
		- Khatri et al. expects a double underscore at the end of each entry
	- Padding for symmetrical arrays that `numpy` needed. So, switched to Pytorch models. Hopefully, quantum models will also not have this issue.
 	- Expected issue, as no one has used spanish embeddings.
  - Big picture- end-to-end model for Uspantekan and Spanish
- Notes for meeting with Robert
-  Current baselines: with about 100 sentences as input
	- Spanish:
 		- without embeddings, with 100 sentences, classification accuracy is 71%. Very tunable
 		- Next plan: see what classification accuracy we get with embeddings
   	- Uspantekan
   		- Uspantekan data, no embeddings
   	 	- Classification accuracy: 72%
- Next steps:
	- see what classification accuracy we get with Spanish embeddings, to see if embeddings can improve scores. This will help us rely on non-English text
 	- Tuning
  	- Get F1 scores to assess all quadrants of the testing
  	- Assess with quantum computing is able to get us closer to underspecificity.

## August 28th 2024
- Why Spider? It works, gives results- use it for Spanish, as well as uspantekan
	- Use for both cups and ansatz
	-  
- Model
	- Use BCE loss- original code coded it, instead of calling it
 - To Do
 	- v6 code- lambeq and pytorch have version issues
  	- Do `lightning` and `lambeq` work together?
   - try with v6 code and pytorch
    	-getting the symbol doesnt have size issue in laptop. Sounds like a version mismatch between lambeq and pytorch
    	- try on colab
      	- update- gets the same error on colab
       	- try v4 on colab.-update: works fine. 
        - also try v4 on laptop
        - if no error related to pytorch in trainer.fit:
        - 	update@august31st-OOV error, . fixed by taking .fit() out of wandb sweep.v4 ran end to end for both spanish and uspantekan- atleast till trainer.fit since v4 didnt have the second ML model that khatri uses
        - 	find why v6 is not running.
        - 	update@august31st-v6 still giving OOV error in first model
        - 	remember: goal here is to get the spanish to work end to end with spanish embeddings. t
        - 	then we willt hink about aligning with uspantekan translations for OOV in uspantekan
        - 	
- 	else:
        - switch to Quantum Model- using actual quantum computer. If we are fighting stupid infrastructure and dll issues might as well do it for quantum model, not stupid pytorch or numpy models.
  	
## August 26th 2024
1. try with NUMPy model and square matrix issue
	-  try with making all sentences padded with . -- failed -bob cat parser, automatically removed . 
	-  try with same sentence- works fine./no matrix/array difference issue
	-  sentence level: 
	-  diagram level: ignored
	-  circuit level: tried adding dummy circuit. i.e say XX gates back to back ==1 but became a pain since they wanted it in Rx Ry gate. 
		- why was this not a problem earlier
   		- why 10?
- why did this not happen in the english version- or even uspantekan version?- our own code?
	- in khatri tehsiswas he terminating english sentence- go back and look at his original code- answer: no, he is also doing same maxlength <=
- what he is doing with maxlen.- picks sentences less than maxlength. 
2. what are the potential solutions
	- without quantum level change
	- try with period.--failed
	- try with filler words. uhm--failed
	- tokenizer spanish
	- how is our own english/uspantekan code different than the spanish one. Are we using a different spider?
	- spacy tokenizer
3. update. we decided to do this comparison first. i.e compare between v4 (the code which worked end to end for uspantekan) kept [here](https://github.com/ua-datalab/QNLP/blob/mithun_dev/v4_load_uspantekan_using_spider_classical.py) and v6(the code which is not working for spanish) [here] (https://github.com/ua-datalab/QNLP/blob/mithun_dev/v6_qnlp_uspantekan_experiments.py)
	- how is khatri's code different than the spanish one. Are we using a different spider?
	- with quantum level change
4. Replace with quantum model/simulation?
5. once we have end to end system running, to improve accuracy, add a spanish tokenizer expliciity

## August 21st 2024
- Main bug- Lambeq's classical example uses Numpy, which is set up to require square matricies as input.
	- Khatri et al. uses the first 10 words in the sentence, discards the rest.
 	- Code does not run when sentences have >10 words
- Potential Solutions
	- try padding sentences with special characters or filler words?
  		- This did not workwith special characters, which got filtered out
  	- Filler words like "um"
  	- Choose n_words >10, based on the data?
  		- ToDo Megh: work on this
 - Mithun contacted Robert Henderson, requested meeting  

## August 9th 2024
- Quick discussion on the difference between raw embeddings and other options
- ToDo Tuesday: create materials for student presentation, and progress in the QNLP project
- Debugging the code to fix fasstext issues
	- The print statements are printing circuits, making it hard to see which words are the issue
 	- How are OOV words identified and their no. calculated? Is Fasttext being implemented correctly?
	-  not the main issue- embeddings are being used correctly    
  	- Is the code using the fasttext embeddings at all? or the spacy spanish parser?
  	- Updates to code- added a try-except chunk: when a word is really OOV, the code will stop and print details for us
  	- Examine why training loop is utilizing only 14 out of 90 sentences, why is bobcat parser not working with the rest?
  	- Embeddings not being passed in the right format- hence a shape error

## July 30th 2024
- General discussion for wrapping up summer responsibilities- and plan going forward
- Close reading of section 6.1-6.3
- skip gram model vs context-aware methods
- Mapping fasttext embeddings and the given input

## July 25th 2024
- Looked at the Khatri et al. code for Spanish, and worked on code fix: https://github.com/ua-datalab/QNLP/tree/megh_dev
	-  Most of the word are OOV, so they can't be put into grammatical categories
	-   ToDo: Assess what khatri et al. does with OOV cases (section 6)
	-   ToDo: run code on VSCode
- General Discussion:
	-  How does Quantum trainer compare to NN?
 		- Feed forward- we look at the loss between the original and predicted value, does back propagation, until the right combination of weights provides us useable prediction
  		- Instead of neurons, we use quantum circuits
    		- Instead of parameters for gates, we have a different system
      		- Objective- minimize loss, and get optimal assignment of parameters, best combo of weights, to find mapping between input setence and label. "Why does a group pf words belong to x category, not y"?
        	- What is loss value?
        		- Classical: diff between gold and predicted, we backprop
          		-  QNLP- Instead of weights, think of angles. machine's job is to find optimal "angles"  
  		- Thus, the trainer for QC is the same as NN- both have similar black boxes
    			- Very hard to explain the difference between two sentences with the same syntax, but different meanings.
      		- Different words will have different angles- so we can explain semantic differences in syntactically identical sentences
		- The code should look the same for deep and quantum trainer
- What is "OOV", from a coding perspective?
	- out of vocabulary words are assigned the same weights, which is not an accurate way to proceed
	- Fast text, Byte Pair Encoding: ways to solve this problem by using embeddings (n-gram models) to assign different weights to different OOV words
 		- Our code needs to learn some kind of mapping between ngrams and angles  
	- In our code, nearly all words are assigned OOV labels

## June 30th 2024
- Discussion of Khatri et al., section 6
	- Continuous bag of words model- teaching a machine a simple pattern matching method. When X word exists, works Y, Z are also likely to occur.
 	- "meaning is context", not "context is meaning".
  	- Word embeddings have a similar understanding of mental lexicon as expeiments on lexical semantics?
  	- Fast text- improving OOV issues by either working with n-grams (thus, meaning agnostic)
  	- Mithun's idea- use GPT embeddings which may take care of OOV words by assessign words in a network of related words, rather than related phones.
  	- 6.2- words belonging to one topic or category will be seen together. Two models are used, a general embedding, as well as a perceptron trained on task-specific vocabulary
  	- Implementation of the baseline models- deeper models performed better than surface models
  - ToDo- look at 6.0-6.4 again, and come back with notes. 

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
 	- Mithun shared his updated code for khatri et. al., that works on Uspantekan:  https://github.com/ua-datalab/QNLP/blob/mithun_dev/v4_load_uspantekan_using_spider_classical.py
  - 
  - Overhauled code to fit our classification task that has only one feature vector, as opposed to two. (that is because khatri code was designed for NLI kind of tasks, which expects a pair like hypotehsis and premise)+ `lambeq` libraries and modules needed to be replaced due to depreciation.

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
	- Filename non-ASCII issue resolved with rename   and train, test, dev splits saved
	- Classical case-  [ran into error while running sentence2diagram](https://www.google.com/url?q=https://colab.research.google.com/drive/12kNxLNX162hGznIYenBSqLJbflmFaE1y?usp%3Dsharing&sa=D&source=editors&ust=1717607867019672&usg=AOvVaw1SAvjipfXEAOkwHKcnRCgQ)
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
		+  0 for bailar, 1 for educacion
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
		+  0 for bailar, 1 for educacion
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
	1. Set up and Try out lambeq classification [task](https://www.google.com/url?q=https://cqcl.github.io/lambeq/tutorials/trainer-classical.html&sa=D&source=editors&ust=1717607867023254&usg=AOvVaw3Se6n9TAlMX5oyEHCO2C1Z)  on Cyverse
	2. With spanish text from the Robert henderson’s data
		1. Pick 2 classes (i.e file names in the data directory)
			1. dancing/bailes
			2. [Education](https://www.google.com/url?q=https://github.com/bkeej/usp_qnlp/blob/main/data/UD/Acontecimientos_sobrenaturales.conllu&sa=D&source=editors&ust=1717607867023895&usg=AOvVaw3gzoN1EoQiznA9p15PbxWz)
			3. Try to recreate the same ML pipeline shown in the lambeq task above.
				1. Using spacy spanish tokenizer.
3. Concrete steps:
	1. Download code from the Lambeq tutorial’s   [Quantum](https://www.google.com/url?q=https://cqcl.github.io/lambeq/tutorials/trainer-quantum.html&sa=D&source=editors&ust=1717607867024391&usg=AOvVaw0hSG13wlkmHZP5akrEMoPD)  case
	2. Replace [training](https://www.google.com/url?q=https://github.com/CQCL/lambeq/blob/main/docs/examples/datasets/rp_train_data.txt&sa=D&source=editors&ust=1717607867024688&usg=AOvVaw0C4Tf2Ane5m0bQGPVI7Mj4)  data from relative clauses to the text classification task (see ‘Classical Case’ [https://cqcl.github.io/lambeq/tutorials/trainer-classical.html](https://www.google.com/url?q=https://cqcl.github.io/lambeq/tutorials/trainer-classical.html&sa=D&source=editors&ust=1717607867024954&usg=AOvVaw0iw7KHxbXYMsbUCXhl0ft6)
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
* Lambeq- pip install lambeq  to install
* If we know the minima of the gradient descent- can we build language up from it?
* TODO- install lambeq  and feed it a sentence \[done on Cyverse\]
* Run end to end- work on it like a tutorial
* Think- tokenizer available for English, Spanish, but not other languages. How do we work without one?
	- Run this on Spanish first
	- Think of a problem
* Jupyter notebook stored at: /data-store/iplant/home/mkrishnaswamy/qnlp

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
- why did we decide to go with spanish first and not Uspanthekan?
	- Lambeq pipeline seems to have a language model requirements and needs embeddings. We have some for Spanish, none for Uspantekan
	- We have direct Uspanteqan-Spanish translations, but not English-Uspanteqan. Which means that if things fail, we have no way to examine what happened if we used an English model.
- How many hours can Megh work on QNLP?
	- This semester: 5hrs, as workshop is also a responsibility
 	- Two 10hr projects may derail dissertation
- Spring 2025: 12/8 split will work, as fewer hours needed for creating content    
