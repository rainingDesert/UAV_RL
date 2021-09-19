# Demo for VAE Optimization

## Pre-request
-Python 3.9
-Numpy
-Pytorch 1.9.0
-jenkins
-jenkinsapi
-tqdm

## Train Autoencoder and Approximate Reward Predictor (Only Driven Parameters) (New version no longer support)

User could mark the second command in **run_train.bat** and only run the first one. We provide some simulated data in **dataset/data.csv**. Trained weight of autoencoder and reward network are store in **store/after_reward.pth** and **store/ae.pth**. In the demo, reward is set as "max_hover_time_s".

## Train Autoencoder for both Driven Parameters and Component Instance

Run the second command in **run_train.bat**. As no prior knowledge to know what makes a valid composition of parameters, data to train autoencoder are randomly sampled.

## Train Reinforcement Learning for both Driven Parameters and Component Instance

Run the third command in **run_train.bat**. As no prior knowledge to know what makes a valid composition of parameters, data to train RL are randomly sampled. Information about whether different designs are valid are stored in file **full_record.csv**, which could be used to get rid of some simulation.

## Bayesian Optimization to Optimize Only Driven Parameters

The first command in "run_optim.bat" optimize driven parameters with autoencoder to reduce dimension of search space and approximate reward network to predict reward.

## Bayesian Optimization to Optimize both Driven Parameters and Component Instance

The second command in "run_optim.bat" optimize driven parameters and component instance. Similar as above, autoencoder is used to encode two kinds of parameters into continuous latent space. While jenkins is leverage for simulation.

## Random Sample Test

Run the third command in "run_optim.bat".

## Demo Results
Search space of component instance in Demo.

|Component| Candidate Instance|
|:--|:--|
|Battery|TurnigyGraphene1000mAh2S75C, TurnigyGraphene1000mAh4S75C, TurnigyGraphene1300mAh4S75C, TurnigyGraphene1200mAh6S75C, TurnigyGraphene1400mAh3S75C, TurnigyGraphene1500mAh3S75C, TurnigyGraphene1600mAh4S75C, TurnigyGraphene2200mAh3S75C|
|ESC|ESC_debugging, kde_direct_KDEXF_UAS20LV, kde_direct_KDEXF_UAS55, t_motor_AIR_20A, t_motor_ALPHA_40A, t_motor_AT_40A,t_motor_AT_115A|
|Propeller|apc_propellers_6x4EP, apc_propellers_6x6EP, apc_propellers_7x4EP, apc_propellers_7x5EP, apc_propellers_8x8EP|
|Motor|t_motor_AS2312KV1400, t_motor_AT2321KV1250, t_motor_MN22041400KV, t_motor_MN4010KV475, t_motor_MT13063100KV|

### Best Parameters

Component Instance:
|Component| Best Instance|
|:--|:--|
|Battery|TurnigyGraphene2200mAh3S75C|
|ESC|t_motor_AIR_20A|
|Propeller|apc_propellers_6x4EP|
|Motor|t_motor_AT2321KV1250|

Driven Parameters:
|Parameter|Value|
|:--|--:|
|ArmLength|200.047|
|SupportLength|121.610|
|BatteryOffset_X|-0.087|
|BatteryOffset_z|-0.656|