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

# Consistent Attr

CONSISTENT_ATTR_PROMPT_TEMPLATE_Q1 = "The provided image arranges key frames from an AI generated video in a grid layout.  Describe the video, carefully examining objects rendering quality throughout the frames and their visual attributes."

CONSISTENT_ATTR_PROMPT_TEMPLATE_Q2 = "Please select one option from A to E for the question. \n \
Question: \n \
A: '{phrase_1}' is clearly portrayed in all the frames. \n \
B: '{phrase_1}' is present in some frames. \n \
C: '{phrase_1}' is not strictly portrayed (mix other feature). \n \
D: '{phrase_1}' is incorrectly portrayed (wrong feature). \n \
E: '{phrase_1}' is not present at all. \n \
Here is an example: the question evaluates the rendering of 'A green taxi'. If the green taxi is clearly absent in more than half frames, select B. If the taxi is both green and yellow, select C. If the taxi is not green, select D. If there is no taxi at all, select E.\n \
Select the most suitable option according to your previous description. \
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C)."

CONSISTENT_ATTR_PROMPT_TEMPLATE_Q3 = "Please select one option from A to E for the question. \n \
Question: \n \
A: '{phrase_2}' is clearly portrayed in all the frames. \n \
B: '{phrase_2}' is present in some frames. \n \
C: '{phrase_2}' is not strictly portrayed (mix other feature). \n \
D: '{phrase_2}' is incorrectly portrayed (wrong feature). \n \
E: '{phrase_2}' is not present at all. \n \
Here is an example: the question evaluates the rendering of 'A green taxi'. If the green taxi is clearly absent in more than half frames, select B. If the taxi is both green and yellow, select C. If the taxi is not green, select D. If there is no taxi at all, select E.\n \
Select the most suitable option according to your previous description. \
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C)."

# Dynamic Attr

DYNAMIC_ATTR_PROMPT_TEMPLATE_Q1 = "The provided image arranges 6 key frames from an AI generated video in a 3-row by 2-column grid layout.  Describe the video, focusing on the interactions between the characters or objects that appear throughout the frames."

DYNAMIC_ATTR_PROMPT_TEMPLATE_Q2 = "To evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please carefully answer the following question.\n \
Question: \n \
A: All the objects involved in the interaction are clearly portrayed in the video. \n \
B: Some objects involved in the interaction are not depicted very clear. \n \
C: Some objects involved in the interaction are missing. \n \
D: None of the objects involved in the interaction are present. \n \
Select the most suitable option according to the video and your previous description. \
Provide your answer in JSON format with the following keys: option (e.g., B), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., A)."

DYNAMIC_ATTR_PROMPT_TEMPLATE_Q3_A = "To evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please carefully examine the interaction and select the most suitable option.\n \
A: The interaction process is clearly shown, with the objects involved engaging in a dynamic manner. The outcome logically follow from the preceding actions and aligns accurately with what the prompt indicated, if mentioned. \n \
B: The interaction process is mostly clear, with objects engaging actively. The outcome generally aligns with what the prompt indicated, if mentioned. \n \
C: The interaction process is somewhat clear, but the objects show limited engagement. The development of process might be unclear, and while the outcome aligns with the prompt but with no previous actions. \n \
D: The interaction process is unclear, with minimal engagement from the objects. There is little to no development of the process. The outcome, if mentioned in prompt, is vague. \n \
E: The interaction process is virtually nonexistent, with no visible engagement from the objects. The outcome in the prompt, if mentioned, is absent or completely irrelevant.\n \
Provide your answer in a JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."

DYNAMIC_ATTR_PROMPT_TEMPLATE_Q3_B = "Following the previous question, to evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please select the most suitable option.\n \
A: the interaction of the clearly presented object(s) is highly dynamic and compensates effectively for the unclear ones, with a logical outcome that implies the interaction. \n \
B: the interaction of the clearly presented object(s) is weak, with confusing relationship to the unclear ones and an irrelevant outcome to the prompt.  \n \
Provide your answer in a JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."

DYNAMIC_ATTR_PROMPT_TEMPLATE_Q3_C = "Following the previous question, to evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please select the most suitable option.\n \
A: the interaction of the clearly presented object(s) is highly dynamic and compensates effectively for the missing ones, with a logical outcome that implies the interaction. \n \
B: the interaction of the clearly presented object(s) is weak, with confusing relationship to the missing ones and an irrelevant outcome to the prompt.  \n \
Provide your answer in a JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."
