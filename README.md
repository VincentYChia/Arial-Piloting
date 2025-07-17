A modification of Typefly (https://github.com/typefly/TypeFly) focusing on localizing all LLM calls while still maintaining intelligent piloting capabilities.


Primary changes occured in the LLM_wrapper, LLM_controller, and LLM_planner.
The architecture has also been noticably changed- with there being an additional LLM call before Minispec writing occurs to a reasoning model. 
