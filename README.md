## Contest Structure

1. Participants are allowed to take on three tasks at the same time, but they cannot play for both
"attack" and "defence" teams for the same task.

2. We have Grey box attack scenario. The attack team has knowledge about the input data on which defence
models are trained on. They can run gradient based update for generating adversarial perturbations. However,
they will have no information about the Defence technique used by the other team.

3. Every submission will be evaluated against several baseline methods provided by the
organizers, as well as the models submitted by the other participants playing for the opposite
team.

4. Every participant will only have a limited amount of submissions; this will prevent participants
from trying to guess our test labels.


### Guidelines

1. TUM employees cannot participate in the contest.
2. You must register for the contest with one valid email address. If you participate in the contest with multiple email addresses, you will be disqualified.
3. The registration times are listed in the schedule section. If you register after the time periods, your evaluation will not be considered.
4. The contest is divided into four separate tasks, Targeted Face Re-Identification (Attacks and Defence) and Face Attribute Alterations(Attacks and Defence). Participants can be part of either the Attack track or Defence track but not both. However, participants for the Attack track of Face Attribute Alteration can compete for the Defence track of Targeted Face Re-Identification (but not for Defence track of Face Attribute Alteration), and vice versa.
5. Participants are required to release the code of their submissions as open-source to our cloud systems.
6. A model submitted by the defence team should be capable of handling valid inputs. In case the model fails to classify the input due to some error in the model, we penalize the model by considering it as a failure to handle an adversarial example.
7. The Attack team should produce an adversarial example in the gray-box setting. They can use model gradients to create adversarial perturbations. If the execution fails for a valid input example, due to some logical errors in the submission, we penalize the Attack team by skipping that example and assuming that the defence team was able to handle the perturbation without any misclassification.
8. Although we use a hidden test set for evaluation, still each classifier must be stateless. It should not memorize the training images.
9. The defence models should not produce random outputs for the same input at different points of time. Participants should make sure that their models are deterministic.
10. The number of submissions is limited to three times throughout the contest period (we will select the best submission from the three).
11. We have 1000 test samples which should be perturbed within 4 hours. If the time duration is exceeded, the perturbation process is interrupted and the team is penalized.
12. The Attack teams will be ranked based on the amount of decrease in accuracy they cause. This ranking will be based on the highest value first. defence teams will be ranked based on the decrease in their accuracy. The team which has maximum decrease will be placed last while the team having a minimum decrease in accuracy is ranked first. A separate ranking will be prepared for each of the four sub-tasks.

## Procedure

### Initial Round
For the contest, we will have two teams, one concerned with attacks while the other with defence.
##### Attack
For the attack submissions, we will have an initial round. Here, we will use a model trained based on Adversarial training using FGSM attacks. This model would be pitted against the attack-submissions. We will evaluate the decrease in accuracy of the model based on the attack. We refer to this decrease as the <b> delta </b> value. For instance, if the original accuracy of the model on the task was 80% and the perturbed samples decreased it to 60%, the delta score is 20.
 
Based on this delta value, the top 5 submissions would be selected. This would be our finalists for the attack configuration.
##### Defence
Similarly, for the defence teams, we would evaluate the model based on FGSM, PGD and BIM attack methods. Here we follow similar ideas but the idea would be to evaluate how robust the model is based on how little a decrease in accuracy it suffers. For instance if there are 3 submissions with <b>delta</b> values of 10, 2, 5, we would select the team with a delta value of 2. Since we are running three attacks, we take a weighted sum of the decrease with FGSM having 0.2 weight coeff and BIM and PGD having values of 0.4 and 0.4 respectively.

### Final Round
Finally, we have the top 5 attack teams and top 5 defence teams. We will pitch them against each other. 

Each defence team submissions would be run against the 5 attack teams. We select the best defence based on the model which **suffers the minimum accuracy decrease overall**.

For selecting the best attack method, we would select the attack method which causes **maximum decrease in accuracy of the defence models overall**.

## Some Details

1. We have an maximum allowed perturbation of epsilon = 8/255.
2. We use L-inf norm for all the submissions
3. We use 2 Nvidia GPUs (Titan X Pascal and GeForce GTX) with 12 GB VRAM. We have a 32 core AMD CPU with 128 GB RAM. The code shall be run using docker containers which would use Ubuntu:18.04 base image and cuda:10.1 with access to both the available GPUs.
3. The adversarial sample generation process has an upper limit time of 4 hours (240 minutes) for 1000 samples.


We check for all the above mentioned requirements for computing the delta value. For instance, in our `attacks` folder we have created an instance of CarliniWagnerL2 attack. Since L-inf < L2 , the samples generated would invariably violate condition 1 and as such, would be penalized. Please refer to the file `Adversarial.py` to see the steps in action. 
    
### Getting Started

Please take a look [here](dev_toolkit/startup_guide.md) for details about the Development Toolkit structure and start up guide.   

### Submission Steps

Please take a look [here](dev_toolkit/submission_steps.md) for details about your model submission.