## Competition Structure

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

1. TUM employees can participate but are ineligible for prizes.
2. To be eligible for the final scoring, participants are required to release the code of their
submissions as open source.
3. Any legitimate input that is not classified by a model (e.g. for which an error is produced) will
be counted as an adversarial.
4. If an attack fails to produce a misclassification of the sample (e.g. because it produces an
error), then we will register a worst-case adversarial instead (a uniform grey image).
5. Each classifier must be stateless and act one image at a time. This rule is supposed to
prevent strategies such as memorizing pre-attack images and classifying replayed versions
of them at defence time.
6. The decision of each classifier must be deterministic. In other words, the classifier decision
must be the same for the same input at any point in time.
7. The number of submissions is limited to **three** times (we will select the best submission from the three).

## Procedure

### Initial Round
For the competition, we will have two teams, one concerned with attacks while the other with defence.
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
3. The adversarial sample generation process has an upper limit time of 256 minutes for batch size of 64


We check for all the above mentioned requirements for computing the delta value. For instance, in our `attacks` folder we have created an instance of CarliniWagnerL2 attack. Since L-inf < L2 , the samples generated would invariably violate condition 1 and as such, would be penalized. Please refer to the file `Adversarial.py` to see the steps in action. 
    
### Getting Started

Please take a look [here](dev_toolkit/startup_guide.md) for details about the Development Toolkit structure and start up guide.   

### Submission Steps

Please take a look [here](dev_toolkit/submission_steps.md) for details about your model submission.