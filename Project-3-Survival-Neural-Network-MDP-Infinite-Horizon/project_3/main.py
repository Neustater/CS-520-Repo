import agent_1, agent_2, agent_3, agent_4, agent_u_star, agent_v_star, agent_u_partial, agent_v_partial, agent_u_partial_v_star
import prey
import predator
import environment
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import u_star
import dill

def main():

	num_runs = 3000
	num_environments = 1
	total_runs = num_runs * num_environments

	a1_caught = 0
	a1_died = 0
	a1_time_out = 0
	a1_steps = 0

	a2_caught = 0
	a2_died = 0
	a2_time_out = 0
	a2_steps = 0

	a3_caught = 0
	a3_died = 0
	a3_time_out = 0
	a3_steps = 0
	a3_prey_certain = 0

	a4_caught = 0
	a4_died = 0
	a4_time_out = 0
	a4_steps = 0
	a4_prey_certain = 0

	au_caught = 0
	au_died = 0
	au_time_out = 0
	au_steps = 0

	av_caught = 0
	av_died = 0
	av_time_out = 0
	av_steps = 0

	aup_caught = 0
	aup_died = 0
	aup_time_out = 0
	aup_steps = 0
	aup_prey_certain = 0

	avp_caught = 0
	avp_died = 0
	avp_time_out = 0
	avp_steps = 0
	avp_prey_certain = 0

	auvp_caught = 0
	auvp_died = 0
	auvp_time_out = 0
	auvp_steps = 1
	auvp_prey_certain = 0

	with open("u_star_env.pkl", 'rb') as file:
		loaded_tuple = dill.load(file)
	
	input_environment, env_u_star = loaded_tuple

	with open("v_star_env.pkl", 'rb') as file:
		env_v_star = dill.load(file)

	with open("v_partial_env.pkl", 'rb') as file:
		env_v_partial = dill.load(file)
		
	for j in range(num_runs):
		print(f"{j} ", end="", flush=True)

		input_predator = predator.Predator()
		input_prey = prey.Prey() 
		input_pos = random.choice(range(0,49))

		#make sure agent doesnt start in occupied node
		while input_prey.pos == input_pos or input_predator.pos == input_pos:
			input_pos = random.choice(range(0,49))
		
		test_agent_1 = agent_1.Agent_1(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos)
		k = test_agent_1.move()
		a1_steps += k[1]
		result_1 = k[0]
		if result_1 == 1:
			a1_caught += 1
		elif result_1 == 0:
			a1_died += 1
		elif result_1 == -1:
			a1_time_out +=1
		
		test_agent_2 = agent_2.Agent_2(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos)
		k = test_agent_2.move()
		a2_steps += k[1]
		result_2 = k[0]
		if result_2 == 1:
			a2_caught += 1
		elif result_2 == 0:
			a2_died += 1
		elif result_2 == -1:
			a2_time_out +=1
		
		test_agent_3 = agent_3.Agent_3(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos)
		k = test_agent_3.move()
		a3_steps += k[1]
		result_3 = k[0]
		if result_3 == 1:
			a3_caught += 1
		elif result_3 == 0:
			a3_died += 1
		elif result_3 == -1:
			a3_time_out +=1
		a3_prey_certain += test_agent_3.certain_prey_pos
		
		test_agent_4 = agent_4.Agent_4(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos)
		k = test_agent_4.move()
		a4_steps += k[1]
		result_4 = k[0]
		if result_4 == 1:
			a4_caught += 1
		elif result_4 == 0:
			a4_died += 1
		elif result_4 == -1:
			a4_time_out +=1
		a4_prey_certain += test_agent_4.certain_prey_pos
		
		test_agent_u = agent_u_star.Agent_u_star(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos, env_u_star = env_u_star)
		k = test_agent_u.move()
		au_steps += k[1]
		result_u = k[0]
		if result_u == 1:
			au_caught += 1
		elif result_u == 0:
			au_died += 1
		elif result_u == -1:
			au_time_out +=1
		
		test_agent_v = agent_v_star.Agent_v_star(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos, env_v_star = env_v_star)
		k = test_agent_v.move()
		av_steps += k[1]
		result_v = k[0]
		if result_v == 1:
			av_caught += 1
		elif result_v == 0:
			av_died += 1
		elif result_v == -1:
			av_time_out +=1

		test_agent_up = agent_u_partial.Agent_u_partial(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos, env_u_star = env_u_star)
		k = test_agent_up.move()
		aup_steps += k[1]
		result_up = k[0]
		if result_up == 1:
			aup_caught += 1
		elif result_up == 0:
			aup_died += 1
		elif result_up == -1:
			aup_time_out +=1
		aup_prey_certain += test_agent_up.certain_prey_pos

		test_agent_vp = agent_v_partial.Agent_v_partial(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos, env_v_partial = env_v_partial)
		k = test_agent_vp.move()
		avp_steps += k[1]
		result_vp = k[0]
		if result_vp == 1:
			avp_caught += 1
		elif result_vp == 0:
			avp_died += 1
		elif result_vp == -1:
			avp_time_out +=1
		avp_prey_certain += test_agent_vp.certain_prey_pos
		
		test_agent_uvp = agent_u_partial_v_star.Agent_u_partial_v_star(copy.deepcopy(input_predator), copy.deepcopy(input_prey), copy.deepcopy(input_environment), input_pos, env_v_star = env_v_star, options_matrix = env_u_star.options_matrix)
		k = test_agent_uvp.move()
		auvp_steps += k[1]
		result_uvp = k[0]
		if result_uvp == 1:
			auvp_caught += 1
		elif result_uvp == 0:
			auvp_died += 1
		elif result_uvp == -1:
			auvp_time_out +=1
		auvp_prey_certain += test_agent_uvp.certain_prey_pos

			
	print()
			
	print("\nAgent 1:")
	print(f"Caught (including timeout): {round((a1_caught/total_runs) * 100, 3)}% | Died (including timeout): {round((a1_died/total_runs) * 100, 3)}% | Timed Out %: {round((a1_time_out/total_runs) * 100, 3)}%")
	print(f"Caught (excluding timeout): {round((a1_caught/(total_runs-a1_time_out)) * 100, 3)}% | Died (exlcuding) timeout): {round((a1_died/(total_runs-a1_time_out) * 100),3)}% | Avg Steps: {a1_steps/total_runs}")

	print("\nAgent 2:")
	print(f"Caught (including timeout): {round((a2_caught/total_runs) * 100, 3)}% | Died (including timeout): {round((a2_died/total_runs) * 100, 3)}% | Timed Out %: {round((a2_time_out/total_runs) * 100, 3)}%")
	print(f"Caught (excluding timeout): {round((a2_caught/(total_runs-a2_time_out)) * 100, 3)}% | Died (exlcuding) timeout): {round((a2_died/(total_runs-a2_time_out) * 100),3)}% | Avg Steps: {a2_steps/total_runs}")

	print("\nAgent U*:")
	print(f"Caught (including timeout): {round((au_caught/total_runs) * 100, 3)}% | Died (including timeout): {round((au_died/total_runs) * 100, 3)}% | Timed Out %: {round((au_time_out/total_runs) * 100, 3)}%")
	print(f"Caught (excluding timeout): {round((au_caught/(total_runs-au_time_out)) * 100, 3)}% | Died (exlcuding) timeout): {round((au_died/(total_runs-au_time_out) * 100),3)}% | Avg Steps: {au_steps/total_runs}")

	print("\nAgent V*:")
	print(f"Caught (including timeout): {round((av_caught/total_runs) * 100, 3)}% | Died (including timeout): {round((av_died/total_runs) * 100, 3)}% | Timed Out %: {round((av_time_out/total_runs) * 100, 3)}%")
	print(f"Caught (excluding timeout): {round((av_caught/(total_runs-av_time_out)) * 100, 3)}% | Died (exlcuding) timeout): {round((av_died/(total_runs-av_time_out) * 100),3)}% | Avg Steps: {av_steps/total_runs}")


	print("\nAgent 3:")
	print(f"Caught (including timeout): {round((a3_caught/total_runs) * 100, 3)}% | Died (including timeout): {round((a3_died/total_runs) * 100, 3)}% | Timed Out %: {round((a3_time_out/total_runs) * 100, 3)}%")
	print(f"Caught (excluding timeout): {round((a3_caught/(total_runs-a3_time_out)) * 100, 3)}% | Died (exlcuding) timeout): {round((a3_died/(total_runs-a3_time_out) * 100),3)}% | Avg Steps: {a3_steps/total_runs}")
	print(f"Steps where certain of prey pos (and correct): {round(a3_prey_certain/a3_steps * 100, 3)}%")

	print("\nAgent 4:")
	print(f"Caught (including timeout): {round((a4_caught/total_runs) * 100, 3)}% | Died (including timeout): {round((a4_died/total_runs) * 100, 3)}% | Timed Out %: {round((a4_time_out/total_runs) * 100, 3)}%")
	print(f"Caught (excluding timeout): {round((a4_caught/(total_runs-a4_time_out)) * 100, 3)}% | Died (exlcuding) timeout): {round((a4_died/(total_runs-a4_time_out) * 100),3)}% | Avg Steps: {a4_steps/total_runs}")
	print(f"Steps where certain of prey pos (and correct): {round(a4_prey_certain/a4_steps * 100, 3)}%")

	print("\nAgent U Partial:")
	print(f"Caught (including timeout): {round((aup_caught/total_runs) * 100, 3)}% | Died (including timeout): {round((aup_died/total_runs) * 100, 3)}% | Timed Out %: {round((aup_time_out/total_runs) * 100, 3)}%")
	print(f"Caught (excluding timeout): {round((aup_caught/(total_runs-aup_time_out)) * 100, 3)}% | Died (exlcuding) timeout): {round((aup_died/(total_runs-aup_time_out) * 100),3)}% | Avg Steps: {aup_steps/total_runs}")
	print(f"Steps where certain of prey pos (and correct): {round(aup_prey_certain/aup_steps * 100, 3)}%")

	print("\nAgent V Partial:")
	print(f"Caught (including timeout): {round((avp_caught/total_runs) * 100, 3)}% | Died (including timeout): {round((avp_died/total_runs) * 100, 3)}% | Timed Out %: {round((avp_time_out/total_runs) * 100, 3)}%")
	print(f"Caught (excluding timeout): {round((avp_caught/(total_runs-avp_time_out)) * 100, 3)}% | Died (exlcuding) timeout): {round((avp_died/(total_runs-avp_time_out) * 100),3)}% | Avg Steps: {avp_steps/total_runs}")
	print(f"Steps where certain of prey pos (and correct): {round(avp_prey_certain/avp_steps * 100, 3)}%")

	print("\nAgent U V* Partial:")
	print(f"Caught (including timeout): {round((auvp_caught/total_runs) * 100, 3)}% | Died (including timeout): {round((auvp_died/total_runs) * 100, 3)}% | Timed Out %: {round((auvp_time_out/total_runs) * 100, 3)}%")
	print(f"Caught (excluding timeout): {round((auvp_caught/(total_runs-auvp_time_out)) * 100, 3)}% | Died (exlcuding) timeout): {round((auvp_died/(total_runs-auvp_time_out) * 100),3)}% | Avg Steps: {auvp_steps/total_runs}")
	print(f"Steps where certain of prey pos (and correct): {round(auvp_prey_certain/auvp_steps * 100, 3)}%")


	caught_data = {'Agent 1':round((a1_caught/total_runs) * 100, 3), 
	'Agent 2':round((a2_caught/total_runs) * 100, 3),
	'Agent U*':round((au_caught/total_runs) * 100, 3),
	'Agent V*':round((av_caught/total_runs) * 100, 3),
	'Agent 3':round((a3_caught/total_runs) * 100, 3), 
	'Agent 4':round((a4_caught/total_runs) * 100, 3),
	'Agent U Partial':round((aup_caught/total_runs) * 100, 3),
	'Agent V Partial':round((avp_caught/total_runs) * 100, 3),
	'Agent U V* Partial':round((auvp_caught/total_runs) * 100, 3)}

	death_data = {'Agent 1':round((a1_died/total_runs) * 100, 3),
	'Agent 2':round((a2_died/total_runs) * 100, 3),
	'Agent U*':round((au_died/total_runs) * 100, 3),
	'Agent V*':round((av_died/total_runs) * 100, 3),
	'Agent 3':round((a3_died/total_runs) * 100, 3),
	'Agent 4':round((a4_died/total_runs) * 100, 3),
	'Agent U Partial':round((aup_died/total_runs) * 100, 3),
	'Agent V Partial':round((avp_died/total_runs) * 100, 3),
	'Agent U V* Partial':round((auvp_died/total_runs) * 100, 3)}

	time_out_data = {'Agent 1':round((a1_time_out/total_runs) * 100, 3),
	'Agent 2':round((a2_time_out/total_runs) * 100, 3),
	'Agent U*':round((au_time_out/total_runs) * 100, 3),
	'Agent V*':round((av_time_out/total_runs) * 100, 3),
	'Agent 3':round((a3_time_out/total_runs) * 100, 3),
	'Agent 4':round((a4_time_out/total_runs) * 100, 3),
	'Agent U Partial':round((aup_time_out/total_runs) * 100, 3),
	'Agent V Partial':round((avp_time_out/total_runs) * 100, 3),
	'Agent U V* Partial':round((auvp_time_out/total_runs) * 100, 3)}

	steps_data = {'Agent 1':(a1_steps/total_runs),
	'Agent 2':(a2_steps/total_runs),
	'Agent U*':(au_steps/total_runs),
	'Agent V*':(av_steps/total_runs),
	'Agent 3':(a3_steps/total_runs),
	'Agent 4':(a4_steps/total_runs),
	'Agent U Partial':(aup_steps/total_runs),
	'Agent V Partial':(avp_steps/total_runs),
	'Agent U V* Partial':(auvp_steps/total_runs)}

	prey_certainty_data = {
		'Agent 3':round(a3_prey_certain/a3_steps * 100, 3),
		'Agent 4':round(a4_prey_certain/a4_steps * 100, 3),
		'Agent U Partial':round(aup_prey_certain/aup_steps * 100, 3),
		'Agent V Partial':round(avp_prey_certain/avp_steps * 100, 3),
		'Agent U V* Partial':round(auvp_prey_certain/auvp_steps * 100, 3)}
			 
	figs_size = (10,7)
	
	Catch_rates = list(caught_data.values())
	Death_rates = list(death_data.values())
	Time_Out_rates = list(time_out_data.values())
	Steps_rates = list(steps_data.values())
	Agent_names = list(caught_data.keys())
	Prey_certain_agents = list(prey_certainty_data.keys())
	Prey_certain = list(prey_certainty_data.values())
	## Plots for complete information setting

	fig_complete_info = plt.figure(figsize = figs_size)

	plt.bar(Agent_names[:4], Catch_rates[:4], color ='green')
	plt.bar(Agent_names[:4], Death_rates[:4], color ='maroon')
	plt.bar(Agent_names[:4], Time_Out_rates[:4], color ='yellow')
	colors = {'Deaths':'maroon', 'Captures':'green', 'Timeouts':'yellow'}         
	labels = list(colors.keys())
	handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
	plt.legend(handles, labels)
	plt.xlabel("Agents")
	plt.ylabel("Percentage of Occurence for Complete Information Setting")
	plt.title("Rate of Events Causing End Game")
	plt.tight_layout()
	plt.xticks(rotation = 30)
	#plt.show() 
	fig_complete_info.savefig('plots/Plot1_complete_information_setting.png', bbox_inches='tight')

	fig_complete_info_steps = plt.figure(figsize= figs_size)

	plt.bar(Agent_names[:4], Steps_rates[:4], color ='gold')

	plt.xlabel("Agents")
	plt.ylabel("Average Number of Steps")
	plt.title("Comparision of average number of steps for Complete Information setting")
	plt.tight_layout()
	plt.xticks(rotation = 30)
	#plt.show() 

	fig_complete_info_steps.savefig('plots/Plot2_complete_information_setting.png', bbox_inches='tight')


	## Plots for Partial Prey information setting 

	fig_partial_prey = plt.figure(figsize = figs_size)

	plt.bar(Agent_names[4:8], Catch_rates[4:8], color ='green')
	plt.bar(Agent_names[4:8], Death_rates[4:8], color ='maroon')
	plt.bar(Agent_names[4:8], Time_Out_rates[4:8], color ='yellow')
	colors = {'Deaths':'maroon', 'Captures':'green', 'Timeouts':'yellow'}         
	labels = list(colors.keys())
	handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
	plt.legend(handles, labels)
	plt.xlabel("Agents")
	plt.ylabel("Percentage of Occurence")
	plt.title("Rate of Events Causing End Game for Partial Prey Information Setting")
	plt.tight_layout()
	plt.xticks(rotation = 30)
	#plt.show() 
	fig_partial_prey.savefig('plots/Plot1_partial_prey_information_setting.png', bbox_inches='tight')

	fig_partial_prey_steps = plt.figure(figsize= figs_size)

	plt.bar(Agent_names[4:8], Steps_rates[4:8], color ='gold')

	plt.xlabel("Agents")
	plt.ylabel("Average Number of Steps")
	plt.title("Comparision of average number of steps for Partial Prey Information setting")
	plt.tight_layout()
	plt.xticks(rotation = 30)
	#plt.show() 

	fig_partial_prey_steps.savefig('plots/Plot2_partial_prey_information_setting.png', bbox_inches='tight')

	#Plots for U and V Partials

	fig_partial_prey = plt.figure(figsize = figs_size)

	plt.bar(Agent_names[6:9], Catch_rates[6:9], color ='green')
	plt.bar(Agent_names[6:9], Death_rates[6:9], color ='maroon')
	plt.bar(Agent_names[6:9], Time_Out_rates[6:9], color ='yellow')
	colors = {'Deaths':'maroon', 'Captures':'green', 'Timeouts':'yellow'}         
	labels = list(colors.keys())
	handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
	plt.legend(handles, labels)
	plt.xlabel("Agents")
	plt.ylabel("Percentage of Occurence")
	plt.title("Rate of Events Causing End Game for U and V Partials")
	plt.tight_layout()
	plt.xticks(rotation = 30)
	#plt.show() 
	fig_partial_prey.savefig('plots/Plot1_U_V_partial_prey.png', bbox_inches='tight')

	fig_partial_prey_steps = plt.figure(figsize= figs_size)

	plt.bar(Agent_names[6:9], Steps_rates[6:9], color ='gold')

	plt.xlabel("Agents")
	plt.ylabel("Average Number of Steps")
	plt.title("Comparision of average number of steps for U and V Partials")
	plt.tight_layout()
	plt.xticks(rotation = 30)
	#plt.show() 

	fig_partial_prey_steps.savefig('plots/Plot2_U_V_partial_prey.png', bbox_inches='tight')

	## All Plot
	
	fig_all = plt.figure(figsize = figs_size) # Decrease dpi to get higher resolution
	plt.bar(Agent_names, Catch_rates, color ='green')
	plt.bar(Agent_names, Death_rates, color ='maroon')
	plt.bar(Agent_names, Time_Out_rates, color ='yellow')
	colors = {'Deaths':'maroon', 'Captures':'green', 'Timeouts':'yellow'}         
	labels = list(colors.keys())
	handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
	plt.legend(handles, labels)
	plt.xlabel("Agents")
	plt.ylabel("Percentage of Occurence")
	plt.title("Rate of Events Causing End Game")
	plt.tight_layout()
	plt.xticks(rotation = 30)
	fig_all.savefig('plots/Plot1_all.png', bbox_inches='tight')


	## MISC plots

	figs = plt.figure(figsize = figs_size) # Decrease dpi to get higher resolution
	
	plt.xlabel("Agents")
	plt.ylabel("Percentage of Runs Agent Catches Prey")
	plt.title("Agent Catches")
	plt.bar(Agent_names, Catch_rates, color ='green')
	plt.tight_layout()
	plt.xticks(rotation = 30)
	#plt.show() 
	figs.savefig('plots/Success_Plot.png', bbox_inches='tight')

	figs = plt.figure(figsize = figs_size)

	plt.xlabel("Agents")
	plt.ylabel("Percentage of Runs Agent Dies")
	plt.title("Agent Deaths")
	plt.bar(Agent_names, Death_rates, color ='maroon')
	plt.tight_layout()
	plt.xticks(rotation = 30)
	#plt.show() 
	figs.savefig('plots/Failure_Plot.png', bbox_inches='tight')

	figs = plt.figure(figsize = figs_size)

	plt.xlabel("Agents")
	plt.ylabel("Average Steps taken by Agent")
	plt.title("Steps taken by Agents")
	plt.bar(Agent_names, Steps_rates, color ='gold')
	plt.tight_layout()
	plt.xticks(rotation = 30)
	#plt.show() 
	figs.savefig('plots/Avg_steps_plot.png', bbox_inches='tight')

	figs = plt.figure(figsize = figs_size)

	plt.xlabel("Agents")
	plt.ylabel("Percentage")
	plt.title("Percentage of Steps Certain of Prey Position (and Correct)")
	plt.bar(Prey_certain_agents, Prey_certain, color ='purple')
	plt.tight_layout()
	plt.xticks(rotation = 30)
	#plt.show() 
	figs.savefig('plots/Prey_certainity_Plot.png', bbox_inches='tight')


	
	



if __name__ == '__main__':
	main()