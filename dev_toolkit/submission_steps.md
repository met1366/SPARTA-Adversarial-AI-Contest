In this tutorial, we'll cover the process of submitting your results for official evaluation.
Once your model has been evaluated officially, your scores will be added to the leaderboard.

Before we begin, please register yourself by sending an email to **register_aicontest@sec.in.tum.de**. 

Please clearly mention whether you are participating in the Defence Attack track along with the subtask (Attribute
Alteration or Targeted ReIdentification) you are participating in.

### Evaluating on the test set

To preserve the integrity of test results, we do not release the test set labels to the public.
Instead, we require you to send your code and pre-trained model weights so that we can run it on the test set for you.

If you plan to have your model officially evaluated, please plan 2 weeks in advance to allow sufficient time for your model
results to be on the leaderboard. Because we use automated evaluation to evaluate your model, please follow the steps exactly
to avoid further delays in evaluation.

##### Step 1: Upload trained model with source code 
We'll download your source folder for the trained model along with the pretrained model weights.
If you are participating in only the Attacks, you do not need to upload model weights. It is only for
teams participating for the Defence track.


##### Step 2: Building Docker Image
We would build docker image with the source code and model weights. We would mount the hidden test set
data on the Docker containers. 

Please take a look at the readme for information about Docker image used in the project. We used the
nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 Docker image which has CUDA support. Participants can use different packages
as needed specific to the project. Please make sure your Dockerfile is up to date and not having bugs.

##### Step 3: Run the Model on Test Set
In the next step we execute the model and compute scores as mentioned in the Rules. We will upload the scores
on the Leader Dashboard. 

#### Naming convention
While sending the model, please follow the naming convention
`<team_name>.<task>.<dataset>.<attempt>`. For instance, if this is your first submission, team name is test, you are an attack team and 
are working on the ReIdentification task, you would send the model name 
```
test.attack.reid.1
``` 


#### Submitting multiple models
It's common to submit multiple versions of the model. While sending the model, please follow the above mentioned naming convention.
This enables us to keep track of your model easily. For instance, continuing on the same example given above, if you want to submit
a new version of the model ie v2, you would need to send:- 
```
test.attack.reid.2
``` 

If this model performs better than the previous one on our hidden test set, we would update the leader-board accordingly. Once the processing
of your model is finished, you would also get an email with details of the execution and score. If the new model has a lower score than the older
one, we still consider your older model score for the leader-board.

Please do not submit more than 2 models within a day. We use a FIFO based idea for evaluation hence the time it takes for evaluating your submission
would depend on the congestion. We recommend to not delay the submissions for the deadline. 


#### Changing the Model name
Since the model names are closely related to the team name, please stick with the naming convention. Also, please be very specific with the **attempt numbers**.
If we receive two submissions with the same **attempt** number, we would randomly select one for the execution but you would lose one attempt.
 