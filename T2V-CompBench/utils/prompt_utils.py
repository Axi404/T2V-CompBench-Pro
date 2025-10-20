# Action Binding
ACTION_BINDING_PROMPT_TEMPLATE_Q1 = "The provided image arranges key frames from an AI generated video in a grid layout.  Describe the video, highlight all the characters and objects that appear throughout the frames and indicate how they act."

ACTION_BINDING_PROMPT_TEMPLATE_Q2 = "To evaluate if the text '{this_prompt}' is correctly portrayed in the video, please carefully answer the following questions. \n \
Question: \n \
A: Both '{obj1}' and {obj2} are clearly present in the video. \n \
B: Only {obj1} is present, {obj2} is not depicted \n \
C: Only {obj2} is present, {obj1} is not depicted \n \
D: Neither {obj1} nor {obj2} appears in the video. \
Select the most suitable option according to the video and your previous description. \
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C)."

ACTION_BINDING_PROMPT_TEMPLATE_Q3_A = "Please select the most suitable options for the two questions: \n \
Question 1: \n\
A1: '{obj1_action}' is clearly depicted . \n \
B1: It is not obvious if '{obj1_action}'. \n \
C1: The action of '{obj1_action}' is not depicted \n \
Question 2: \n\
A2: '{obj2_action}' is clearly depicted . \n \
B2: It is not obvious if '{obj2_action}'. \n \
C2: The action of '{obj2_action}' is not depicted \n \
Put each option in a JSON format with the following keys: option (e.g., A1,B2), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., A1,C2)."

ACTION_BINDING_PROMPT_TEMPLATE_Q3_BC_OBJ1 = "Please select the most suitable option:\n \
A: '{obj1_action}' is clearly depicted . \n \
B: It is not obvious if '{obj1_action}'. \n \
C: The action of '{obj1_action}' is not depicted \n \
Put the options in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."

ACTION_BINDING_PROMPT_TEMPLATE_Q3_BC_OBJ2 = "Please select the most suitable option:\n \
A: '{obj2_action}' is clearly depicted . \n \
B: It is not obvious if '{obj2_action}'. \n \
C: The action of '{obj2_action}' is not depicted \n \
Put the options in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."