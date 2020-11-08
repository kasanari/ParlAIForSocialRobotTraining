# The main implementation of my thesis work
This is a forked version of ParlAI which I used for my master's thesis. The main change is the addition of a DialoGPT agent, and tasks to train agents on the social robot data.
For more details on the project, please refer to my [master's thesis report](https://github.com/kasanari/ParlAIForSocialRobotTraining/blob/dialogpt/thesis_report_jakob_nyberg_final.pdf).
## Example

### Input:
The human and the robot will begin playing the game soon. The robot wants to get to know the human player better. They have already talked about playing games, but the robot has not asked the human player which games he/she enjoys playing yet. If the robot is neutral, how would it react to the human player liking the same type of games, and keep a natural conversation going?

### Cherry picked outputs from different decoders, using dialogpt with multitasking trained on both empathetic dialouges and robot data:

Greedy: I like to play games too.

Nucleus: Well I do enjoy those games, do you?

Top-k: i have heard very good things about this game can you tell me more about it
