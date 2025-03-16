// Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
// SPDX-License-Identifier: MIT

import * as asyncio from "asyncio";
import * as copy from "copy";
import * as functools from "functools";
import * as inspect from "inspect";
import * as json from "json";
import * as logging from "logging";
import * as re from "re";
import * as warnings from "warnings";
import { defaultdict } from "collections";
import { contextmanager } from "contextlib";
import { dataclass } from "dataclasses";
import { signature } from "inspect";
import { AbstractCache } from "..cache.cache.js";
import { PYTHON_VARIANTS, UNKNOWN, check_can_use_docker_or_throw, content_str, decide_use_docker, execute_code, extract_code, infer_lang } from "..code_utils.js";
import { CodeExecutor } from "..coding.base.js";
import { CodeExecutorFactory } from "..coding.factory.js";
import { InvalidCarryOverTypeError, SenderRequiredError } from "..exception_utils.js";
import { IOStream } from "..io.base.js";
import { LLMConfig, LLMConfigFilter } from "..llm_config.js";
import { ClearConversableAgentHistoryMessage, ClearConversableAgentHistoryWarningMessage, ConversableAgentUsageSummaryMessage, ConversableAgentUsageSummaryNoCostIncurredMessage, ExecuteCodeBlockMessage, ExecuteFunctionMessage, ExecutedFunctionMessage, GenerateCodeExecutionReplyMessage, TerminationAndHumanReplyNoInputMessage, TerminationMessage, UsingAutoReplyMessage, create_received_message_model } from "..messages.agent_messages.js";
import { ModelClient, OpenAIWrapper } from "..oai.client.js";
import { log_event, log_function_use, log_new_agent, logging_enabled } from "..runtime_logging.js";
import { ChatContext, Tool, load_basemodels_if_needed, serialize_to_str } from "..tools.js";
import { Agent, LLMAgent } from ".agent.js";
import { ChatResult, _post_process_carryover_item, a_initiate_chats, initiate_chats } from ".chat.js";
import { consolidate_chat_info, gather_usage_summary } from ".utils.js";
import { z } from 'zod';

const logger = logging.getLogger(__name__);

const F = z.function();

// Parameter name for context variables
// Use the value in functions and they will be substituted with the context variables:
// e.g. function myFunction(contextVariables: Record<string, any>, myOtherParameters: any): any {}
export const __CONTEXT_VARIABLES_PARAM_NAME__ = "context_variables";

class Dataclass {}


class UpdateSystemMessage {
    /**
     * Update the agent's system message before they reply
     *
     * Args:
     *     content_updater: The format string or function to update the agent's system message. Can be a format string or a Callable.
     *         If a string, it will be used as a template and substitute the context variables.
     *         If a Callable, it should have the signature:
     *             def my_content_updater(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> str
     */

    content_updater: Callable | string;

    constructor(content_updater: Callable | string) {
        this.content_updater = content_updater;
        this.__post_init__();
    }

    __post_init__() {
        if (typeof this.content_updater === 'string') {
            // find all {var} in the string
            const vars = this.content_updater.match(/\{(\w+)\}/g);
            if (!vars || vars.length === 0) {
                console.warn("Update function string contains no variables. This is probably unintended.");
            }

        } else if (typeof this.content_updater === 'function') {
            const sig = this.content_updater.length;
            if (sig !== 2) {
                throw new Error(
                    "The update function must accept two parameters of type ConversableAgent and List<Dict<string, any>>, respectively"
                );
            }
            // Assuming return type check is done elsewhere as TypeScript does not have runtime type checks
        } else {
            throw new Error("The update function must be either a string or a callable");
        }
    }
}

class UPDATE_SYSTEM_MESSAGE extends UpdateSystemMessage {
    /** Deprecated: Use UpdateSystemMessage instead. This class will be removed in a future version (TBD). */

    constructor(...args: any[]) {
        warnings.warn(
            "UPDATE_SYSTEM_MESSAGE is deprecated and will be removed in a future version (TBD). Use UpdateSystemMessage instead.",
            DeprecationWarning,
            { stacklevel: 2 }
        );
        super(...args);
    }
}


class ConversableAgent extends LLMAgent {
    /**
    (In preview) A class for generic conversable agents which can be configured as assistant or user proxy.

    After receiving each message, the agent will send a reply to the sender unless the msg is a termination msg.
    For example, AssistantAgent and UserProxyAgent are subclasses of this class,
    configured with different default settings.

    To modify auto reply, override `generate_reply` method.
    To disable/enable human response in every turn, set `human_input_mode` to "NEVER" or "ALWAYS".
    To modify the way to get human input, override `get_human_input` method.
    To modify the way to execute code blocks, single code block, or function call, override `execute_code_blocks`,
    `run_code`, and `execute_function` methods respectively.
    */

    static DEFAULT_CONFIG: false | Record<string, any> = false;  // False or dict, the default config for llm inference
    static MAX_CONSECUTIVE_AUTO_REPLY = 100;  // maximum number of consecutive auto replies (subject to future change)

    static DEFAULT_SUMMARY_PROMPT = "Summarize the takeaway from the conversation. Do not add any introductory phrases.";
    static DEFAULT_SUMMARY_METHOD = "last_msg";
    llm_config: Record<string, any> | false;


    constructor(
        name: string,
        system_message: string | string[] = "You are a helpful AI Assistant.",
        is_termination_msg?: (msg: { [key: string]: any }) => boolean,
        max_consecutive_auto_reply?: number,
        human_input_mode: "ALWAYS" | "NEVER" | "TERMINATE" = "TERMINATE",
        function_map?: { [key: string]: (...args: any[]) => any },
        code_execution_config: { [key: string]: any } | false = false,
        llm_config?: LLMConfig | { [key: string]: any } | false,
        llm_config_filter?: LLMConfigFilter | { [key: string]: any },
        default_auto_reply: string | { [key: string]: any } = "",
        description?: string,
        chat_messages?: { [key: string]: { [key: string]: any }[] },
        silent?: boolean,
        context_variables?: { [key: string]: any },
        functions?: ((...args: any[]) => any)[] | ((...args: any[]) => any),
        update_agent_state_before_reply?: (
            | ((...args: any[]) => any)[]
            | ((...args: any[]) => any)
        )
    ) {
        /**
        Args:
            name (str): name of the agent.
            system_message (str or list): system message for the ChatCompletion inference.
            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                When set to 0, no auto reply will be generated.
            human_input_mode (str): whether to ask for human inputs every time a message is received.
                Possible values are "ALWAYS", "TERMINATE", "NEVER".
                (1) When "ALWAYS", the agent prompts for human input every time a message is received.
                    Under this mode, the conversation stops when the human input is "exit",
                    or when is_termination_msg is True and there is no human input.
                (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
                    the number of auto reply reaches the max_consecutive_auto_reply.
                (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
                    when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.
            function_map (dict[str, callable]): Mapping function names (passed to openai) to callable functions, also used for tool calls.
            code_execution_config (dict or False): config for the code execution.
                To disable code execution, set to False. Otherwise, set to a dictionary with the following keys:
                - work_dir (Optional, str): The working directory for the code execution.
                    If None, a default working directory will be used.
                    The default working directory is the "extensions" directory under
                    "path_to_autogen".
                - use_docker (Optional, list, str or bool): The docker image to use for code execution.
                    Default is True, which means the code will be executed in a docker container. A default list of images will be used.
                    If a list or a str of image name(s) is provided, the code will be executed in a docker container
                    with the first image successfully pulled.
                    If False, the code will be executed in the current environment.
                    We strongly recommend using docker for code execution.
                - timeout (Optional, int): The maximum execution time in seconds.
                - last_n_messages (Experimental, int or str): The number of messages to look back for code execution.
                    If set to 'auto', it will scan backwards through all messages arriving since the agent last spoke, which is typically the last time execution was attempted. (Default: auto)
            llm_config (LLMConfig or dict or False or None): llm inference configuration.
                Please refer to [OpenAIWrapper.create](/docs/api-reference/autogen/OpenAIWrapper#autogen.OpenAIWrapper.create)
                for available options.
                When using OpenAI or Azure OpenAI endpoints, please specify a non-empty 'model' either in `llm_config` or in each config of 'config_list' in `llm_config`.
                To disable llm-based auto reply, set to False.
                When set to None, will use self.DEFAULT_CONFIG, which defaults to False.
            llm_config_filter (LLMConfigFilter or dict): llm config filter to filter the llm config.
                It can be a dict or an instance of LLMConfigFilter.
            default_auto_reply (str or dict): default auto reply when no code execution or llm-based reply is generated.
            description (str): a short description of the agent. This description is used by other agents
                (e.g. the GroupChatManager) to decide when to call upon this agent. (Default: system_message)
            chat_messages (dict or None): the previous chat messages that this agent had in the past with other agents.
                Can be used to give the agent a memory by providing the chat history. This will allow the agent to
                resume previous had conversations. Defaults to an empty chat history.
            silent (bool or None): (Experimental) whether to print the message sent. If None, will use the value of
                silent in each function.
            context_variables (dict or None): Context variables that provide a persistent context for the agent.
                Note: Will maintain a reference to the passed in context variables (enabling a shared context)
                Only used in Swarms at this stage:
                https://docs.ag2.ai/docs/reference/agentchat/contrib/swarm_agent
            functions (List[Callable[..., Any]]): A list of functions to register with the agent.
                These functions will be provided to the LLM, however they won't, by default, be executed by the agent.
                If the agent is in a swarm, the swarm's tool executor will execute the function.
                When not in a swarm, you can have another agent execute the tools by adding them to that agent's function_map.
            update_agent_state_before_reply (List[Callable[..., Any]]): A list of functions, including UpdateSystemMessage's, called to update the agent before it replies.
        */
        // we change code_execution_config below and we have to make sure we don't change the input
        // in case of UserProxyAgent, without this we could even change the default value {}
        code_execution_config =
            code_execution_config && typeof code_execution_config === 'object' && 'copy' in code_execution_config
                ? { ...code_execution_config }
                : code_execution_config;

        // a dictionary of conversations, default value is list
        if (chat_messages === undefined) {
            this._oai_messages = new Map<string, any[]>();
        } else {
            this._oai_messages = chat_messages;
        }

        this._oai_system_message = [{ content: system_message, role: "system" }];
        this._description = description !== undefined ? description : system_message;
        this._is_termination_msg =
            is_termination_msg !== undefined
                ? is_termination_msg
                : (x) => content_str(x["content"]) === "TERMINATE";
        this.silent = silent;
        this.run_executor = undefined;

        // Take a copy to avoid modifying the given dict
        if (llm_config && typeof llm_config === 'object') {
            try {
                llm_config = JSON.parse(JSON.stringify(llm_config));
            } catch (e) {
                throw new TypeError(
                    "Please implement __deepcopy__ method for each value class in llm_config to support deepcopy." +
                        " Refer to the docs for more details: https://docs.ag2.ai/docs/topics/llm_configuration#adding-http-client-in-llm-config-for-proxy",
                    { cause: e }
                );
            }
        }

        this._llm_config_filter =
            llm_config_filter && typeof llm_config_filter === 'object'
                ? new LLMConfigFilter(llm_config_filter)
                : llm_config_filter;
        this.llm_config = this._apply_llm_config_filter(
            this._validate_llm_config(llm_config),
            this._llm_config_filter
        );
        this.client = this._create_client(this.llm_config);
        this._validate_name(name);
        this._name = name;

        if (logging_enabled()) {
            log_new_agent(this, { name, system_message, is_termination_msg, max_consecutive_auto_reply, human_input_mode, function_map, code_execution_config, llm_config, llm_config_filter, default_auto_reply, description, chat_messages, silent, context_variables, functions, update_agent_state_before_reply });
        }

        // Initialize standalone client cache object.
        this.client_cache = null;

        this.human_input_mode = human_input_mode;
        this._max_consecutive_auto_reply =
            max_consecutive_auto_reply !== undefined
                ? max_consecutive_auto_reply
                : this.MAX_CONSECUTIVE_AUTO_REPLY;
        this._consecutive_auto_reply_counter = new Map<string, number>();
        this._max_consecutive_auto_reply_dict = new Map<string, number>();
        this._function_map =
            function_map === undefined
                ? {}
                : Object.fromEntries(
                      Object.entries(function_map).filter(([name, callable]) => this._assert_valid_name(name))
                  );
        this._default_auto_reply = default_auto_reply;
        this._reply_func_list = [];
        this._human_input = [];
        this.reply_at_receive = new Map<string, boolean>();
        this.register_reply([Agent, null], ConversableAgent.generate_oai_reply);
        this.register_reply([Agent, null], ConversableAgent.a_generate_oai_reply, true);

        this._context_variables = context_variables !== undefined ? context_variables : {};

        this._tools = [];

        // Register functions to the agent
        if (Array.isArray(functions)) {
            if (!functions.every((func) => typeof func === 'function')) {
                throw new TypeError("All elements in the functions list must be callable");
            }
            this._add_functions(functions);
        } else if (typeof functions === 'function') {
            this._add_single_function(functions);
        } else if (functions !== undefined) {
            throw new TypeError("Functions must be a callable or a list of callables");
        }

        // Setting up code execution.
        // Do not register code execution reply if code execution is disabled.
        if (code_execution_config !== false) {
            // If code_execution_config is None, set it to an empty dict.
            if (code_execution_config === undefined) {
                console.warn(
                    "Using None to signal a default code_execution_config is deprecated. " +
                        "Use {} to use default or False to disable code execution."
                );
                code_execution_config = {};
            }
            if (typeof code_execution_config !== 'object') {
                throw new Error("code_execution_config must be a dict or False.");
            }

            // We have got a valid code_execution_config.
            this._code_execution_config = code_execution_config;

            if (this._code_execution_config["executor"] !== undefined) {
                if ("use_docker" in this._code_execution_config) {
                    throw new Error(
                        "'use_docker' in code_execution_config is not valid when 'executor' is set. Use the appropriate arg in the chosen executor instead."
                    );
                }

                if ("work_dir" in this._code_execution_config) {
                    throw new Error(
                        "'work_dir' in code_execution_config is not valid when 'executor' is set. Use the appropriate arg in the chosen executor instead."
                    );
                }

                if ("timeout" in this._code_execution_config) {
                    throw new Error(
                        "'timeout' in code_execution_config is not valid when 'executor' is set. Use the appropriate arg in the chosen executor instead."
                    );
                }

                // Use the new code executor.
                this._code_executor = CodeExecutorFactory.create(this._code_execution_config);
                this.register_reply([Agent, null], ConversableAgent._generate_code_execution_reply_using_executor);
            } else {
                // Legacy code execution using code_utils.
                let use_docker = this._code_execution_config["use_docker"];
                use_docker = decide_use_docker(use_docker);
                check_can_use_docker_or_throw(use_docker);
                this._code_execution_config["use_docker"] = use_docker;
                this.register_reply([Agent, null], ConversableAgent.generate_code_execution_reply);
            }
        } else {
            // Code execution is disabled.
            this._code_execution_config = false;
        }

        this.register_reply([Agent, null], ConversableAgent.generate_tool_calls_reply);
        this.register_reply([Agent, null], ConversableAgent.a_generate_tool_calls_reply, true);
        this.register_reply([Agent, null], ConversableAgent.generate_function_call_reply);
        this.register_reply([Agent, null], ConversableAgent.a_generate_function_call_reply, true);
        this.register_reply([Agent, null], ConversableAgent.check_termination_and_human_reply);
        this.register_reply([Agent, null], ConversableAgent.a_check_termination_and_human_reply, true);

        // Registered hooks are kept in lists, indexed by hookable method, to be called in their order of registration.
        // New hookable methods should be added to this list as required to support new agent capabilities.
        this.hook_lists = {
            process_last_received_message: [],
            process_all_messages_before_reply: [],
            process_message_before_send: [],
            update_agent_state: [],
        };

        // Associate agent update state hooks
        this._register_update_agent_state_before_reply(update_agent_state_before_reply);
    }

    private _validate_name(name: string): void {
        if (!this.llm_config || !("config_list" in this.llm_config) || this.llm_config["config_list"].length === 0) {
            return;
        }

        const config_list = this.llm_config["config_list"];
        // The validation is currently done only for openai endpoints
        // (other ones do not have the issue with whitespace in the name)
        if ("api_type" in config_list[0] && config_list[0]["api_type"] !== "openai") {
            return;
        }

        // Validation for name using regex to detect any whitespace
        if (/\s/.test(name)) {
            throw new Error(`The name of the agent cannot contain any whitespace. The name provided is: '${name}'`);
        }
    }

    private _get_display_name(): string {
        /**
        Get the string representation of the agent.

        If you would like to change the standard string representation for an
        instance of ConversableAgent, you can point it to another function.
        In this example a function called _swarm_agent_str that returns a string:
        agent._get_display_name = MethodType(_swarm_agent_str, agent)
        */
        return this.name;
    }

    public toString(): string {
        return this._get_display_name();
    }

    private _add_functions(func_list: ((...args: any[]) => any)[]): void {
        /**
        Add (Register) a list of functions to the agent

        Args:
            func_list (list[Callable[..., Any]]): A list of functions to register with the agent.
        */
        for (const func of func_list) {
            this._add_single_function(func);
        }
    }

    private _add_single_function(func: (...args: any[]) => any, name?: string, description: string = ""): void {
        /**
        Add a single function to the agent

        Args:
            func (Callable): The function to register.
            name (str): The name of the function. If not provided, the function's name will be used.
            description (str): The description of the function, used by the LLM. If not provided, the function's docstring will be used.
        */
        if (name) {
            func._name = name;
        } else if (!("_name" in func)) {
            func._name = func.name;
        }

        if (description) {
            func._description = description;
        } else {
            // Use function's docstring, strip whitespace, fall back to empty string
            func._description = (func.toString().match(/\/\*\*([\s\S]*?)\*\//) || ["", ""])[1].trim();
        }

        // Register the function
        this.register_for_llm({ name, description, silent_override: true })(func);
    }


    _register_update_agent_state_before_reply(
        functions: Optional<list<Callable<...args: any[]>> | Callable<...args: any[]>>
    ) {
        /**
         * Register functions that will be called when the agent is selected and before it speaks.
         * You can add your own validation or precondition functions here.
         *
         * Args:
         *     functions (List[Callable[[], None]]): A list of functions to be registered. Each function
         *         is called when the agent is selected and before it speaks.
         */
        if (functions === null) {
            return;
        }
        if (!Array.isArray(functions) && !(functions instanceof UpdateSystemMessage) && typeof functions !== 'function') {
            throw new Error("functions must be a list of callables");
        }

        if (!Array.isArray(functions)) {
            functions = [functions];
        }

        for (const func of functions) {
            if (func instanceof UpdateSystemMessage) {
                // Wrapper function that allows this to be used in the update_agent_state hook
                // Its primary purpose, however, is just to update the agent's system message
                // Outer function to create a closure with the update function

                function create_wrapper(update_func: UpdateSystemMessage) {

                    function update_system_message_wrapper(
                        agent: ConversableAgent, messages: Array<{ [key: string]: any }>
                    ): Array<{ [key: string]: any }> {
                        let sys_message;
                        if (typeof update_func.content_updater === 'string') {
                            // Templates like "My context variable passport is {passport}" will
                            // use the context_variables for substitution
                            sys_message = OpenAIWrapper.instantiate({
                                template: update_func.content_updater,
                                context: agent._context_variables,
                                allow_format_str_template: true,
                            });
                        } else {
                            sys_message = update_func.content_updater(agent, messages);
                        }

                        agent.update_system_message(sys_message);
                        return messages;
                    }

                    return update_system_message_wrapper;
                }

                this.register_hook({ hookable_method: "update_agent_state", hook: create_wrapper(func) });

            } else {
                this.register_hook({ hookable_method: "update_agent_state", hook: func });
            }
        }
    }

    static _validate_llm_config(
        cls: typeof ConversableAgent, llm_config: Optional<LLMConfig | { [key: string]: any } | false>
    ): LLMConfig | false {
        // if not(llm_config in (None, False) or isinstance(llm_config, [dict, LLMConfig])):
        //     raise ValueError(
        //         "llm_config must be a dict or False or None."
        //     )

        if (llm_config === null) {
            llm_config = LLMConfig.get_current_llm_config();
            if (llm_config === null) {
                llm_config = cls.DEFAULT_CONFIG;
            }
        } else if (typeof llm_config === 'object' && !Array.isArray(llm_config)) {
            llm_config = new LLMConfig(llm_config);
        } else if (llm_config === false || llm_config instanceof LLMConfig) {
            // do nothing
        } else {
            throw new Error("llm_config must be a LLMConfig, dict or False or None.");
        }

        return llm_config;
    }

    static _apply_llm_config_filter(
        cls: typeof ConversableAgent,
        llm_config: LLMConfig | false,
        llm_config_filter: Optional<LLMConfigFilter>,
        exclude: boolean = false,
    ): LLMConfig | false {
        if (llm_config === false) {
            return llm_config;
        }

        return llm_config.apply_filter(llm_config_filter, exclude);
    }

    static _create_client(cls: typeof ConversableAgent, llm_config: LLMConfig | false): Optional<OpenAIWrapper> {
        return llm_config === false ? null : new OpenAIWrapper(llm_config);
    }

    static _is_silent(agent: Agent, silent: Optional<boolean> = false): boolean {
        return agent.silent !== null ? agent.silent : silent;
    }

    get name(): string {
        /** Get the name of the agent. */
        return this._name;
    }

    get description(): string {
        /** Get the description of the agent. */
        return this._description;
    }

    set description(description: string) {
        /** Set the description of the agent. */
        this._description = description;
    }

    get code_executor(): Optional<CodeExecutor> {
        /** The code executor used by this agent. Returns None if code execution is disabled. */
        if (!('_code_executor' in this)) {
            return null;
        }
        return this._code_executor;
    }

    register_reply(
        trigger: typeof Agent | string | Agent | Callable<Agent, boolean> | Array<any>,
        reply_func: Callable,
        position: number = 0,
        config: Optional<any> = null,
        reset_config: Optional<Callable<...args: any[]>> = null,
        ignore_async_in_sync_chat: boolean = false,
        remove_other_reply_funcs: boolean = false,
    ) {
        /** Register a reply function.

        The reply function will be called when the trigger matches the sender.
        The function registered later will be checked earlier by default.
        To change the order, set the position to a positive integer.

        Both sync and async reply functions can be registered. The sync reply function will be triggered
        chats (initiated with `ConversableAgent.a_initiate_chat`). If an `async` reply function is registered
        and a chat is initialized with a sync function, `ignore_async_in_sync_chat` determines the behaviour as follows:
            if `ignore_async_in_sync_chat` is set to `False` (default value), an exception will be raised, and
            if `ignore_async_in_sync_chat` is set to `True`, the reply function will be ignored.

        Args:
            trigger (Agent class, str, Agent instance, callable, or list): the trigger.
                If a class is provided, the reply function will be called when the sender is an instance of the class.
                If a string is provided, the reply function will be called when the sender's name matches the string.
                If an agent instance is provided, the reply function will be called when the sender is the agent instance.
                If a callable is provided, the reply function will be called when the callable returns True.
                If a list is provided, the reply function will be called when any of the triggers in the list is activated.
                If None is provided, the reply function will be called only when the sender is None.
                Note: Be sure to register `None` as a trigger if you would like to trigger an auto-reply function with non-empty messages and `sender=None`.
            reply_func (Callable): the reply function.
                The function takes a recipient agent, a list of messages, a sender agent and a config as input and returns a reply message.

                ```python

                def reply_func(
                    recipient: ConversableAgent,
                    messages: Optional[List[Dict]] = None,
                    sender: Optional[Agent] = None,
                    config: Optional[Any] = None,
                ) -> Tuple[bool, Union[str, Dict, None]]:
                ```
            position (int): the position of the reply function in the reply function list.
                The function registered later will be checked earlier by default.
                To change the order, set the position to a positive integer.
            config (Any): the config to be passed to the reply function.
                When an agent is reset, the config will be reset to the original value.
            reset_config (Callable): the function to reset the config.
                The function returns None. Signature: ```def reset_config(config: Any)```
            ignore_async_in_sync_chat (bool): whether to ignore the async reply function in sync chats. If `False`, an exception
                will be raised if an async reply function is registered and a chat is initialized with a sync
                function.
            remove_other_reply_funcs (bool): whether to remove other reply functions when registering this reply function.
        */
        if (!(typeof trigger === 'function' || typeof trigger === 'string' || trigger instanceof Agent || Array.isArray(trigger))) {
            throw new Error("trigger must be a class, a string, an agent, a callable or a list.");
        }
        if (remove_other_reply_funcs) {
            this._reply_func_list.clear();
        }
        this._reply_func_list.splice(position, 0, {
            trigger: trigger,
            reply_func: reply_func,
            config: config ? { ...config } : null,
            init_config: config,
            reset_config: reset_config,
            ignore_async_in_sync_chat: ignore_async_in_sync_chat && typeof reply_func === 'function' && reply_func.constructor.name === 'AsyncFunction',
        });
    }

    replace_reply_func(old_reply_func: Callable, new_reply_func: Callable) {
        /** Replace a registered reply function with a new one.

        Args:
            old_reply_func (Callable): the old reply function to be replaced.
            new_reply_func (Callable): the new reply function to replace the old one.
        */
        for (const f of this._reply_func_list) {
            if (f.reply_func === old_reply_func) {
                f.reply_func = new_reply_func;
            }
        }
    }

    static _get_chats_to_run(
        chat_queue: Array<{ [key: string]: any }>,
        recipient: Agent,
        messages: Optional<Array<{ [key: string]: any }>>,
        sender: Agent,
        config: any,
    ): Array<{ [key: string]: any }> {
        /** A simple chat reply function.
        This function initiate one or a sequence of chats between the "recipient" and the agents in the
        chat_queue.

        It extracts and returns a summary from the nested chat based on the "summary_method" in each chat in chat_queue.

        Returns:
            Tuple[bool, str]: A tuple where the first element indicates the completion of the chat, and the second element contains the summary of the last chat if any chats were initiated.
        */
        const last_msg = messages[messages.length - 1].content;
        const chat_to_run = [];
        for (let i = 0; i < chat_queue.length; i++) {
            const c = chat_queue[i];
            const current_c = { ...c };
            if (current_c.sender === undefined) {
                current_c.sender = recipient;
            }
            let message = current_c.message;
            // If message is not provided in chat_queue, we by default use the last message from the original chat history as the first message in this nested chat (for the first chat in the chat queue).
            // NOTE: This setting is prone to change.
            if (message === undefined && i === 0) {
                message = last_msg;
            }
            if (typeof message === 'function') {
                message = message(recipient, messages, sender, config);
            }
            // We only run chat that has a valid message. NOTE: This is prone to change depending on applications.
            if (message) {
                current_c.message = message;
                chat_to_run.push(current_c);
            }
        }
        return chat_to_run;
    }

    static _process_nested_chat_carryover(
        chat: { [key: string]: any },
        recipient: Agent,
        messages: Array<{ [key: string]: any }>,
        sender: Agent,
        config: any,
        trim_n_messages: number = 0,
    ): void {
        /** Process carryover messages for a nested chat (typically for the first chat of a swarm)

        The carryover_config key is a dictionary containing:
            "summary_method": The method to use to summarise the messages, can be "all", "last_msg", "reflection_with_llm" or a Callable
            "summary_args": Optional arguments for the summary method

        Supported carryover 'summary_methods' are:
            "all" - all messages will be incorporated
            "last_msg" - the last message will be incorporated
            "reflection_with_llm" - an llm will summarise all the messages and the summary will be incorporated as a single message
            Callable - a callable with the signature: my_method(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> str

        Args:
            chat: The chat dictionary containing the carryover configuration
            recipient: The recipient agent
            messages: The messages from the parent chat
            sender: The sender agent
            config: The LLM configuration
            trim_n_messages: The number of latest messages to trim from the messages list
        */
    }


    static concat_carryover(chat_message: string, carryover_message: string | Array<{ [key: string]: any }>): string {
        /** Concatenate the carryover message to the chat message. */
        const prefix = chat_message ? `${chat_message}\n` : "";

        let content: string;
        if (typeof carryover_message === "string") {
            content = carryover_message;
        } else if (Array.isArray(carryover_message)) {
            content = carryover_message
                .filter(msg => "content" in msg && msg["content"] !== null)
                .map(msg => msg["content"])
                .join("\n");
        } else {
            throw new Error("Carryover message must be a string or a list of dictionaries");
        }

        return `${prefix}Context:\n${content}`;
    }

    carryover_config = chat["carryover_config"];

    if (!("summary_method" in carryover_config)) {
        throw new Error("Carryover configuration must contain a 'summary_method' key");
    }

    const carryover_summary_method = carryover_config["summary_method"];
    const carryover_summary_args = carryover_config["summary_args"] || {};

    let chat_message = "";
    const message = chat.get("message");

    // If the message is a callable, run it and get the result
    if (message) {
        chat_message = typeof message === "function" ? message(recipient, messages, sender, config) : message;
    }

    // deep copy and trim the latest messages
    let content_messages = JSON.parse(JSON.stringify(messages));
    content_messages = content_messages.slice(0, -trim_n_messages);

    let carry_over_message;
    if (carryover_summary_method === "all") {
        // Put a string concatenated value of all parent messages into the first message
        carry_over_message = ConversableAgent.concat_carryover(chat_message, content_messages);
    } else if (carryover_summary_method === "last_msg") {
        carry_over_message = ConversableAgent.concat_carryover(chat_message, content_messages[content_messages.length - 1]["content"]);
    } else if (carryover_summary_method === "reflection_with_llm") {
        chat["recipient"]._oai_messages[sender] = content_messages;

        const carry_over_message_llm = ConversableAgent._reflection_with_llm_as_summary(
            sender,
            chat["recipient"],
            carryover_summary_args
        );

        recipient._oai_messages[sender] = [];

        carry_over_message = ConversableAgent.concat_carryover(chat_message, carry_over_message_llm);
    } else if (typeof carryover_summary_method === "function") {
        const carry_over_message_result = carryover_summary_method(recipient, content_messages, carryover_summary_args);

        carry_over_message = ConversableAgent.concat_carryover(chat_message, carry_over_message_result);
    }

    chat["message"] = carry_over_message;

    static _process_chat_queue_carryover(
        chat_queue: Array<{ [key: string]: any }>,
        recipient: Agent,
        messages: string | ((...args: any[]) => any),
        sender: Agent,
        config: any,
        trim_messages: number = 2
    ): [boolean, string | null] {
        /** Process carryover configuration for the first chat in the queue.

        Args:
            chat_queue: List of chat configurations
            recipient: Receiving agent
            messages: Chat messages
            sender: Sending agent
            config: LLM configuration
            trim_messages: Number of messages to trim for nested chat carryover (default 2 for swarm chats)

        Returns:
            Tuple containing:
                - restore_flag: Whether the original message needs to be restored
                - original_message: The original message to restore (if any)
        */
        let restore_chat_queue_message = false;
        let original_chat_queue_message = null;

        if (chat_queue.length > 0 && "carryover_config" in chat_queue[0]) {
            if ("message" in chat_queue[0]) {
                restore_chat_queue_message = true;
                original_chat_queue_message = chat_queue[0]["message"];
            }

            ConversableAgent._process_nested_chat_carryover(
                chat_queue[0],
                recipient,
                messages,
                sender,
                config,
                trim_messages
            );
        }

        return [restore_chat_queue_message, original_chat_queue_message];
    }

    static _summary_from_nested_chats(
        chat_queue: Array<{ [key: string]: any }>,
        recipient: Agent,
        messages: Array<{ [key: string]: any }> | null,
        sender: Agent,
        config: any
    ): [boolean, string | null] {
        /** A simple chat reply function.
        This function initiate one or a sequence of chats between the "recipient" and the agents in the
        chat_queue.

        It extracts and returns a summary from the nested chat based on the "summary_method" in each chat in chat_queue.

        The first chat in the queue can contain a 'carryover_config' which is a dictionary that denotes how to carryover messages from the parent chat into the first chat of the nested chats). Only applies to the first chat.
            e.g.: carryover_summarize_chat_config = {"summary_method": "reflection_with_llm", "summary_args": None}
            summary_method can be "last_msg", "all", "reflection_with_llm", Callable
            The Callable signature: my_method(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> str
            The summary will be concatenated to the message of the first chat in the queue.

        Returns:
            Tuple[bool, str]: A tuple where the first element indicates the completion of the chat, and the second element contains the summary of the last chat if any chats were initiated.
        */
        const [restore_chat_queue_message, original_chat_queue_message] = ConversableAgent._process_chat_queue_carryover(
            chat_queue, recipient, messages, sender, config
        );

        const chat_to_run = ConversableAgent._get_chats_to_run(chat_queue, recipient, messages, sender, config);
        if (!chat_to_run) {
            return [true, null];
        }
        const res = initiate_chats(chat_to_run);

        if (restore_chat_queue_message) {
            chat_queue[0]["message"] = original_chat_queue_message;
        }

        return [true, res[res.length - 1].summary];
    }

    static async _a_summary_from_nested_chats(
        chat_queue: Array<{ [key: string]: any }>,
        recipient: Agent,
        messages: Array<{ [key: string]: any }> | null,
        sender: Agent,
        config: any
    ): Promise<[boolean, string | null]> {
        /** A simple chat reply function.
        This function initiate one or a sequence of chats between the "recipient" and the agents in the
        chat_queue.

        It extracts and returns a summary from the nested chat based on the "summary_method" in each chat in chat_queue.

        The first chat in the queue can contain a 'carryover_config' which is a dictionary that denotes how to carryover messages from the parent chat into the first chat of the nested chats). Only applies to the first chat.
            e.g.: carryover_summarize_chat_config = {"summary_method": "reflection_with_llm", "summary_args": None}
            summary_method can be "last_msg", "all", "reflection_with_llm", Callable
            The Callable signature: my_method(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> str
            The summary will be concatenated to the message of the first chat in the queue.

        Returns:
            Tuple[bool, str]: A tuple where the first element indicates the completion of the chat, and the second element contains the summary of the last chat if any chats were initiated.
        */
        const [restore_chat_queue_message, original_chat_queue_message] = ConversableAgent._process_chat_queue_carryover(
            chat_queue, recipient, messages, sender, config
        );

        const chat_to_run = ConversableAgent._get_chats_to_run(chat_queue, recipient, messages, sender, config);
        if (!chat_to_run) {
            return [true, null];
        }
        const res = await a_initiate_chats(chat_to_run);
        const index_of_last_chat = chat_to_run[chat_to_run.length - 1]["chat_id"];

        if (restore_chat_queue_message) {
            chat_queue[0]["message"] = original_chat_queue_message;
        }

        return [true, res[index_of_last_chat].summary];
    }

    register_nested_chats(
        chat_queue: Array<{ [key: string]: any }>,
        trigger: typeof Agent | string | Agent | ((agent: Agent) => boolean) | Array<any>,
        reply_func_from_nested_chats: string | ((...args: any[]) => any) = "summary_from_nested_chats",
        position: number = 2,
        use_async: boolean | null = null,
        ...kwargs: any
    ): void {
        /** Register a nested chat reply function.

        Args:
            chat_queue (list): a list of chat objects to be initiated. If use_async is used, then all messages in chat_queue must have a chat-id associated with them.
            trigger (Agent class, str, Agent instance, callable, or list): refer to `register_reply` for details.
            reply_func_from_nested_chats (Callable, str): the reply function for the nested chat.
                The function takes a chat_queue for nested chat, recipient agent, a list of messages, a sender agent and a config as input and returns a reply message.
                Default to "summary_from_nested_chats", which corresponds to a built-in reply function that get summary from the nested chat_queue.
                ```python

                def reply_func_from_nested_chats(
                    chat_queue: List[Dict],
                    recipient: ConversableAgent,
                    messages: Optional[List[Dict]] = None,
                    sender: Optional[Agent] = None,
                    config: Optional[Any] = None,
                ) -> Tuple[bool, Union[str, Dict, None]]:
                ```
            position (int): Ref to `register_reply` for details. Default to 2. It means we first check the termination and human reply, then check the registered nested chat reply.
            use_async: Uses a_initiate_chats internally to start nested chats. If the original chat is initiated with a_initiate_chats, you may set this to true so nested chats do not run in sync.
            kwargs: Ref to `register_reply` for details.
        */
        if (use_async) {
            for (const chat of chat_queue) {
                if (chat["chat_id"] == null) {
                    throw new Error("chat_id is required for async nested chats");
                }
            }
        }

        if (use_async) {
            if (reply_func_from_nested_chats === "summary_from_nested_chats") {
                reply_func_from_nested_chats = this._a_summary_from_nested_chats;
            }
            if (typeof reply_func_from_nested_chats !== "function" || !reply_func_from_nested_chats.constructor.name.includes("AsyncFunction")) {
                throw new Error("reply_func_from_nested_chats must be a callable and a coroutine");
            }

            const wrapped_reply_func = async (recipient: Agent, messages: any = null, sender: Agent = null, config: any = null) => {
                return await reply_func_from_nested_chats(chat_queue, recipient, messages, sender, config);
            };

            functools.update_wrapper(wrapped_reply_func, reply_func_from_nested_chats);

            this.register_reply(
                trigger,
                wrapped_reply_func,
                position,
                kwargs["config"],
                kwargs["reset_config"],
                {
                    ignore_async_in_sync_chat: !use_async ? use_async : kwargs["ignore_async_in_sync_chat"]
                }
            );
        } else {
            if (reply_func_from_nested_chats === "summary_from_nested_chats") {
                reply_func_from_nested_chats = this._summary_from_nested_chats;
            }
            if (typeof reply_func_from_nested_chats !== "function") {
                throw new Error("reply_func_from_nested_chats must be a callable");
            }

            const wrapped_reply_func = (recipient: Agent, messages: any = null, sender: Agent = null, config: any = null) => {
                return reply_func_from_nested_chats(chat_queue, recipient, messages, sender, config);
            };

            functools.update_wrapper(wrapped_reply_func, reply_func_from_nested_chats);

            this.register_reply(
                trigger,
                wrapped_reply_func,
                position,
                kwargs["config"],
                kwargs["reset_config"],
                {
                    ignore_async_in_sync_chat: !use_async ? use_async : kwargs["ignore_async_in_sync_chat"]
                }
            );
        }
    }

    get_context(key: string, defaultValue: any = null): any {
        /** Get a context variable by key.

        Args:
            key: The key to look up
            default: Value to return if key doesn't exist
        Returns:
            The value associated with the key, or default if not found
        */
        return this._context_variables[key] || defaultValue;
    }

    set_context(key: string, value: any): void {
        /** Set a context variable.

        Args:
            key: The key to set
            value: The value to associate with the key
        */
        this._context_variables[key] = value;
    }

    update_context(context_variables: { [key: string]: any }): void {
        /** Update multiple context variables at once.

        Args:
            context_variables: Dictionary of variables to update/add
        */
        Object.assign(this._context_variables, context_variables);
    }


    pop_context(key: string, default: any = null): any {
        /** Remove and return a context variable.

        Args:
            key: The key to remove
            default: Value to return if key doesn't exist
        Returns:
            The value that was removed, or default if key not found
        */
        return this._context_variables.pop(key, default);
    }

    get system_message(): string {
        /** Return the system message. */
        return this._oai_system_message[0]["content"];
    }

    update_system_message(system_message: string): void {
        /** Update the system message.

        Args:
            system_message (str): system message for the ChatCompletion inference.
        */
        this._oai_system_message[0]["content"] = system_message;
    }

    update_max_consecutive_auto_reply(value: number, sender?: Agent): void {
        /** Update the maximum number of consecutive auto replies.

        Args:
            value (int): the maximum number of consecutive auto replies.
            sender (Agent): when the sender is provided, only update the max_consecutive_auto_reply for that sender.
        */
        if (sender === undefined) {
            this._max_consecutive_auto_reply = value;
            for (const k in this._max_consecutive_auto_reply_dict) {
                this._max_consecutive_auto_reply_dict[k] = value;
            }
        } else {
            this._max_consecutive_auto_reply_dict[sender] = value;
        }
    }

    max_consecutive_auto_reply(sender?: Agent): number {
        /** The maximum number of consecutive auto replies. */
        return sender === undefined ? this._max_consecutive_auto_reply : this._max_consecutive_auto_reply_dict[sender];
    }

    get chat_messages(): Record<Agent, Array<Record<string, any>>> {
        /** A dictionary of conversations from agent to list of messages. */
        return this._oai_messages;
    }

    chat_messages_for_summary(agent: Agent): Array<Record<string, any>> {
        /** A list of messages as a conversation to summarize. */
        return this._oai_messages[agent];
    }

    last_message(agent?: Agent): Record<string, any> | null {
        /** The last message exchanged with the agent.

        Args:
            agent (Agent): The agent in the conversation.
                If None and more than one agent's conversations are found, an error will be raised.
                If None and only one conversation is found, the last message of the only conversation will be returned.

        Returns:
            The last message exchanged with the agent.
        */
        if (agent === undefined) {
            const n_conversations = Object.keys(this._oai_messages).length;
            if (n_conversations === 0) {
                return null;
            }
            if (n_conversations === 1) {
                for (const conversation of Object.values(this._oai_messages)) {
                    return conversation[conversation.length - 1];
                }
            }
            throw new Error("More than one conversation is found. Please specify the sender to get the last message.");
        }
        if (!(agent in this._oai_messages)) {
            throw new Error(`The agent '${agent.name}' is not present in any conversation. No history available for this agent.`);
        }
        return this._oai_messages[agent][this._oai_messages[agent].length - 1];
    }

    get use_docker(): boolean | string | null {
        /** Bool value of whether to use docker to execute the code,
        or str value of the docker image name to use, or None when code execution is disabled.
        */
        return this._code_execution_config === false ? null : this._code_execution_config.get("use_docker");
    }

    static _message_to_dict(message: Record<string, any> | string): Record<string, any> {
        /** Convert a message to a dictionary.

        The message can be a string or a dictionary. The string will be put in the "content" field of the new dictionary.
        */
        if (typeof message === 'string') {
            return { "content": message };
        } else if (typeof message === 'object') {
            return message;
        } else {
            return Object(message);
        }
    }

    static _normalize_name(name: string): string {
        /** LLMs sometimes ask functions while ignoring their own format requirements, this function should be used to replace invalid characters with "_".

        Prefer _assert_valid_name for validating user configuration or input
        */
        return name.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
    }

    static _assert_valid_name(name: string): string {
        /** Ensure that configured names are valid, raises ValueError if not.

        For munging LLM responses use _normalize_name to ensure LLM specified names don't break the API.
        */
        if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
            throw new Error(`Invalid name: ${name}. Only letters, numbers, '_' and '-' are allowed.`);
        }
        if (name.length > 64) {
            throw new Error(`Invalid name: ${name}. Name must be less than 64 characters.`);
        }
        return name;
    }

    _append_oai_message(
        message: Record<string, any> | string, role: string, conversation_id: Agent, is_sending: boolean
    ): boolean {
        /** Append a message to the ChatCompletion conversation.

        If the message received is a string, it will be put in the "content" field of the new dictionary.
        If the message received is a dictionary but does not have any of the three fields "content", "function_call", or "tool_calls",
            this message is not a valid ChatCompletion message.
        If only "function_call" or "tool_calls" is provided, "content" will be set to None if not provided, and the role of the message will be forced "assistant".

        Args:
            message (dict or str): message to be appended to the ChatCompletion conversation.
            role (str): role of the message, can be "assistant" or "function".
            conversation_id (Agent): id of the conversation, should be the recipient or sender.
            is_sending (bool): If the agent (aka self) is sending to the conversation_id agent, otherwise receiving.

        Returns:
            bool: whether the message is appended to the ChatCompletion conversation.
        */
        message = this._message_to_dict(message);
        // create oai message to be appended to the oai conversation that can be passed to oai directly.
        const oai_message: Record<string, any> = {
            ...Object.fromEntries(
                Object.entries(message).filter(([k, v]) =>
                    ["content", "function_call", "tool_calls", "tool_responses", "tool_call_id", "name", "context"].includes(k) && v !== null
                )
            )
        };
        if (!("content" in oai_message)) {
            if ("function_call" in oai_message || "tool_calls" in oai_message) {
                oai_message["content"] = null; // if only function_call is provided, content will be set to None.
            } else {
                return false;
            }
        }

        if (["function", "tool"].includes(message["role"])) {
            oai_message["role"] = message["role"];
            if ("tool_responses" in oai_message) {
                for (const tool_response of oai_message["tool_responses"]) {
                    tool_response["content"] = String(tool_response["content"]);
                }
            }
        } else if ("override_role" in message) {
            // If we have a direction to override the role then set the
            // role accordingly. Used to customise the role for the
            // select speaker prompt.
            oai_message["role"] = message["override_role"];
        } else {
            oai_message["role"] = role;
        }

        if (oai_message["function_call"] || oai_message["tool_calls"]) {
            oai_message["role"] = "assistant"; // only messages with role 'assistant' can have a function call.
        } else if (!("name" in oai_message)) {
            // If we don't have a name field, append it
            if (is_sending) {
                oai_message["name"] = this.name;
            } else {
                oai_message["name"] = conversation_id.name;
            }
        }

        this._oai_messages[conversation_id].push(oai_message);

        return true;
    }

    _process_message_before_send(
        message: Record<string, any> | string, recipient: Agent, silent: boolean
    ): Record<string, any> | string {
        /** Process the message before sending it to the recipient. */
        const hook_list = this.hook_lists["process_message_before_send"];
        for (const hook of hook_list) {
            message = hook({
                sender: this, message: message, recipient: recipient, silent: ConversableAgent._is_silent(this, silent)
            });
        }
        return message;
    }

    send(
        message: Record<string, any> | string,
        recipient: Agent,
        request_reply: boolean | null = null,
        silent: boolean | null = false
    ): void {
        /** Send a message to another agent.

        Args:
            message (dict or str): message to be sent.
                The message could contain the following fields:
                - content (str or List): Required, the content of the message. (Can be None)
                - function_call (str): the name of the function to be called.
                - name (str): the name of the function to be called.
                - role (str): the role of the message, any role that is not "function"
                    will be modified to "assistant".
                - context (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](/docs/api-reference/autogen/OpenAIWrapper#autogen.OpenAIWrapper.create).
                    For example, one agent can send a message A as:
        ```python
        {
            "content": lambda context: context["use_tool_msg"],
            "context": {"use_tool_msg": "Use tool X if they are relevant."},
        }
        ```
                    Next time, one agent can send a message B with a different "use_tool_msg".
                    Then the content of message A will be refreshed to the new "use_tool_msg".
                    So effectively, this provides a way for an agent to send a "link" and modify
                    the content of the "link" later.
            recipient (Agent): the recipient of the message.
            request_reply (bool or None): whether to request a reply from the recipient.
            silent (bool or None): (Experimental) whether to print the message sent.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        */
        message = this._process_message_before_send(message, recipient, ConversableAgent._is_silent(this, silent));
        // When the agent composes and sends the message, the role of the message is "assistant"
        // unless it's "function".
        const valid = this._append_oai_message(message, "assistant", recipient, true);
        if (valid) {
            recipient.receive(message, this, request_reply, silent);
        } else {
            throw new Error(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            );
        }
    }

    async a_send(
        message: Record<string, any> | string,
        recipient: Agent,
        request_reply: boolean | null = null,
        silent: boolean | null = false
    ): Promise<void> {
        /** (async) Send a message to another agent.

        Args:
            message (dict or str): message to be sent.
                The message could contain the following fields:
                - content (str or List): Required, the content of the message. (Can be None)
                - function_call (str): the name of the function to be called.
                - name (str): the name of the function to be called.
                - role (str): the role of the message, any role that is not "function"
                    will be modified to "assistant".
                - context (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](/docs/api-reference/autogen/OpenAIWrapper#autogen.OpenAIWrapper.create).
                    For example, one agent can send a message A as:
        ```python
        {
            "content": lambda context: context["use_tool_msg"],
            "context": {"use_tool_msg": "Use tool X if they are relevant."},
        }
        ```
                    Next time, one agent can send a message B with a different "use_tool_msg".
                    Then the content of message A will be refreshed to the new "use_tool_msg".
                    So effectively, this provides a way for an agent to send a "link" and modify
                    the content of the "link" later.
            recipient (Agent): the recipient of the message.
            request_reply (bool or None): whether to request a reply from the recipient.
            silent (bool or None): (Experimental) whether to print the message sent.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        */
        message = this._process_message_before_send(message, recipient, ConversableAgent._is_silent(this, silent));
        // When the agent composes and sends the message, the role of the message is "assistant"
        // unless it's "function".
        const valid = this._append_oai_message(message, "assistant", recipient, true);
        if (valid) {
            await recipient.a_receive(message, this, request_reply, silent);
        } else {
            throw new Error(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            );
        }
    }


    private _print_received_message(message: Record<string, any> | string, sender: Agent, skip_head: boolean = false): void {
        message = this._message_to_dict(message);
        const message_model = create_received_message_model({ message, sender, recipient: this });
        const iostream = IOStream.get_default();
        // message_model.print(iostream.print)
        iostream.send(message_model);
    }

    private _process_received_message(message: Record<string, any> | string, sender: Agent, silent: boolean): void {
        // When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
        const valid = this._append_oai_message(message, "user", sender, false);
        if (logging_enabled()) {
            log_event(this, "received_message", { message, sender: sender.name, valid });
        }

        if (!valid) {
            throw new ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            );
        }

        if (!ConversableAgent._is_silent(sender, silent)) {
            this._print_received_message(message, sender);
        }
    }

    public receive(
        message: Record<string, any> | string,
        sender: Agent,
        request_reply: boolean | null = null,
        silent: boolean | null = false
    ): void {
        /**
         * Receive a message from another agent.
         *
         * Once a message is received, this function sends a reply to the sender or stop.
         * The reply can be generated automatically or entered manually by a human.
         *
         * Args:
         *     message (dict or str): message from the sender. If the type is dict, it may contain the following reserved fields (either content or function_call need to be provided).
         *         1. "content": content of the message, can be None.
         *         2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
         *         3. "tool_calls": a list of dictionaries containing the function name and arguments.
         *         4. "role": role of the message, can be "assistant", "user", "function", "tool".
         *             This field is only needed to distinguish between "function" or "assistant"/"user".
         *         5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
         *         6. "context" (dict): the context of the message, which will be passed to
         *             [OpenAIWrapper.create](/docs/api-reference/autogen/OpenAIWrapper#autogen.OpenAIWrapper.create).
         *     sender: sender of an Agent instance.
         *     request_reply (bool or None): whether a reply is requested from the sender.
         *         If None, the value is determined by `self.reply_at_receive[sender]`.
         *     silent (bool or None): (Experimental) whether to print the message received.
         *
         * Raises:
         *     ValueError: if the message can't be converted into a valid ChatCompletion message.
         */
        this._process_received_message(message, sender, silent);
        if (request_reply === false || (request_reply === null && this.reply_at_receive[sender] === false)) {
            return;
        }
        const reply = this.generate_reply({ messages: this.chat_messages[sender], sender });
        if (reply !== null) {
            this.send(reply, sender, { silent });
        }
    }

    public async a_receive(
        message: Record<string, any> | string,
        sender: Agent,
        request_reply: boolean | null = null,
        silent: boolean | null = false
    ): Promise<void> {
        /**
         * (async) Receive a message from another agent.
         *
         * Once a message is received, this function sends a reply to the sender or stop.
         * The reply can be generated automatically or entered manually by a human.
         *
         * Args:
         *     message (dict or str): message from the sender. If the type is dict, it may contain the following reserved fields (either content or function_call need to be provided).
         *         1. "content": content of the message, can be None.
         *         2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
         *         3. "tool_calls": a list of dictionaries containing the function name and arguments.
         *         4. "role": role of the message, can be "assistant", "user", "function".
         *             This field is only needed to distinguish between "function" or "assistant"/"user".
         *         5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
         *         6. "context" (dict): the context of the message, which will be passed to
         *             [OpenAIWrapper.create](/docs/api-reference/autogen/OpenAIWrapper#autogen.OpenAIWrapper.create).
         *     sender: sender of an Agent instance.
         *     request_reply (bool or None): whether a reply is requested from the sender.
         *         If None, the value is determined by `self.reply_at_receive[sender]`.
         *     silent (bool or None): (Experimental) whether to print the message received.
         *
         * Raises:
         *     ValueError: if the message can't be converted into a valid ChatCompletion message.
         */
        this._process_received_message(message, sender, silent);
        if (request_reply === false || (request_reply === null && this.reply_at_receive[sender] === false)) {
            return;
        }
        const reply = await this.a_generate_reply({ messages: this.chat_messages[sender], sender });
        if (reply !== null) {
            await this.a_send(reply, sender, { silent });
        }
    }

    private _prepare_chat(
        recipient: ConversableAgent,
        clear_history: boolean,
        prepare_recipient: boolean = true,
        reply_at_receive: boolean = true
    ): void {
        this.reset_consecutive_auto_reply_counter(recipient);
        this.reply_at_receive[recipient] = reply_at_receive;
        if (clear_history) {
            this.clear_history(recipient);
            this._human_input = [];
        }
        if (prepare_recipient) {
            recipient._prepare_chat(this, clear_history, false, reply_at_receive);
        }
    }

    private _raise_exception_on_async_reply_functions(): void {
        /**
         * Raise an exception if any async reply functions are registered.
         *
         * Raises:
         *     RuntimeError: if any async reply functions are registered.
         */
        const reply_functions = new Set(
            this._reply_func_list.filter(f => !f.ignore_async_in_sync_chat).map(f => f.reply_func)
        );

        const async_reply_functions = Array.from(reply_functions).filter(f => inspect.iscoroutinefunction(f));
        if (async_reply_functions.length > 0) {
            const msg =
                "Async reply functions can only be used with ConversableAgent.a_initiate_chat(). The following async reply functions are found: " +
                async_reply_functions.map(f => f.name).join(", ");

            throw new RuntimeError(msg);
        }
    }

    public initiate_chat(
        recipient: ConversableAgent,
        clear_history: boolean = true,
        silent: boolean | null = false,
        cache: AbstractCache | null = null,
        max_turns: number | null = null,
        summary_method: string | ((...args: any[]) => any) | null = DEFAULT_SUMMARY_METHOD,
        summary_args: Record<string, any> = {},
        message: Record<string, any> | string | ((...args: any[]) => any) | null = null,
        ...kwargs: any[]
    ): ChatResult {
        /**
         * Initiate a chat with the recipient agent.
         *
         * Reset the consecutive auto reply counter.
         * If `clear_history` is True, the chat history with the recipient agent will be cleared.
         *
         *
         * Args:
         *     recipient: the recipient agent.
         *     clear_history (bool): whether to clear the chat history with the agent. Default is True.
         *     silent (bool or None): (Experimental) whether to print the messages for this conversation. Default is False.
         *     cache (AbstractCache or None): the cache client to be used for this conversation. Default is None.
         *     max_turns (int or None): the maximum number of turns for the chat between the two agents. One turn means one conversation round trip. Note that this is different from
         *         [max_consecutive_auto_reply](#max-consecutive-auto-reply) which is the maximum number of consecutive auto replies; and it is also different from [max_rounds in GroupChat](./groupchat) which is the maximum number of rounds in a group chat session.
         *         If max_turns is set to None, the chat will continue until a termination condition is met. Default is None.
         *     summary_method (str or callable): a method to get a summary from the chat. Default is DEFAULT_SUMMARY_METHOD, i.e., "last_msg".
         *         Supported strings are "last_msg" and "reflection_with_llm":
         *             - when set to "last_msg", it returns the last message of the dialog as the summary.
         *             - when set to "reflection_with_llm", it returns a summary extracted using an llm client.
         *                 `llm_config` must be set in either the recipient or sender.
         *
         *         A callable summary_method should take the recipient and sender agent in a chat as input and return a string of summary. E.g.,
         *
         *         ```python
         *
         *         def my_summary_method(
         *             sender: ConversableAgent,
         *             recipient: ConversableAgent,
         *             summary_args: dict,
         *         ):
         *             return recipient.last_message(sender)["content"]
         *         ```
         *     summary_args (dict): a dictionary of arguments to be passed to the summary_method.
         *         One example key is "summary_prompt", and value is a string of text used to prompt a LLM-based agent (the sender or recipient agent) to reflect
         *         on the conversation and extract a summary when summary_method is "reflection_with_llm".
         *         The default summary_prompt is DEFAULT_SUMMARY_PROMPT, i.e., "Summarize takeaway from the conversation. Do not add any introductory phrases. If the intended request is NOT properly addressed, please point it out."
         *         Another available key is "summary_role", which is the role of the message sent to the agent in charge of summarizing. Default is "system".
         *     message (str, dict or Callable): the initial message to be sent to the recipient. Needs to be provided. Otherwise, input() will be called to get the initial message.
         *         - If a string or a dict is provided, it will be used as the initial message.        `generate_init_message` is called to generate the initial message for the agent based on this string and the context.
         *             If dict, it may contain the following reserved fields (either content or tool_calls need to be provided).
         *
         *                 1. "content": content of the message, can be None.
         *                 2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
         *                 3. "tool_calls": a list of dictionaries containing the function name and arguments.
         *                 4. "role": role of the message, can be "assistant", "user", "function".
         *                     This field is only needed to distinguish between "function" or "assistant"/"user".
         *                 5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
         *                 6. "context" (dict): the context of the message, which will be passed to
         *                     [OpenAIWrapper.create](/docs/api-reference/autogen/OpenAIWrapper#autogen.OpenAIWrapper.create).
         *
         *         - If a callable is provided, it will be called to get the initial message in the form of a string or a dict.
         *             If the returned type is dict, it may contain the reserved fields mentioned above.
         *
         *             Example of a callable message (returning a string):
         *
         *             ```python
         *
         *             def my_message(
         *                 sender: ConversableAgent, recipient: ConversableAgent, context: dict
         *             ) -> Union[str, Dict]:
         *                 carryover = context.get("carryover", "")
         *                 if isinstance(message, list):
         *                     carryover = carryover[-1]
         *                 final_msg = "Write a blogpost." + "\nContext: \n" + carryover
         *                 return final_msg
         *             ```
         *
         *             Example of a callable message (returning a dict):
         *
         *             ```python
         *
         *             def my_message(
         *                 sender: ConversableAgent, recipient: ConversableAgent, context: dict
         *             ) -> Union[str, Dict]:
         *                 final_msg = {}
         *                 carryover = context.get("carryover", "")
         *                 if isinstance(message, list):
         *                     carryover = carryover[-1]
         *                 final_msg["content"] = "Write a blogpost." + "\nContext: \n" + carryover
         *                 final_msg["context"] = {"prefix": "Today I feel"}
         *                 return final_msg
         *             ```
         *     **kwargs: any additional information. It has the following reserved fields:
         *         - "carryover": a string or a list of string to specify the carryover information to be passed to this chat.
         *             If provided, we will combine this carryover (by attaching a "context: " string and the carryover content after the message content) with the "message" content when generating the initial chat
         *             message in `generate_init_message`.
         *         - "verbose": a boolean to specify whether to print the message and carryover in a chat. Default is False.
         *
         * Raises:
         *     RuntimeError: if any async reply functions are registered and not ignored in sync chat.
         *
         * Returns:
         *     ChatResult: an ChatResult object.
         */
        const iostream = IOStream.get_default();

        const _chat_info = { ...locals() };
        _chat_info["sender"] = this;
        consolidate_chat_info(_chat_info, { uniform_sender: this });
        for (const agent of [this, recipient]) {
            agent._raise_exception_on_async_reply_functions();
            agent.previous_cache = agent.client_cache;
            agent.client_cache = cache;
        }
        if (typeof max_turns === "number") {
            this._prepare_chat(recipient, clear_history, false);
            for (let i = 0; i < max_turns; i++) {
                let msg2send;
                if (i === 0) {
                    if (typeof message === "function") {
                        msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs);
                    } else {
                        msg2send = this.generate_init_message(message, ...kwargs);
                    }
                } else {
                    msg2send = this.generate_reply({ messages: this.chat_messages[recipient], sender: recipient });
                }
                if (msg2send === null) {
                    break;
                }
                this.send(msg2send, recipient, { request_reply: true, silent });
            }

            iostream.send(new TerminationMessage({ termination_reason: `Maximum turns (${max_turns}) reached` }));
        } else {
            this._prepare_chat(recipient, clear_history);
            let msg2send;
            if (typeof message === "function") {
                msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs);
            } else {
                msg2send = this.generate_init_message(message, ...kwargs);
            }
            this.send(msg2send, recipient, { silent });
        }
        const summary = this._summarize_chat(
            summary_method,
            summary_args,
            recipient,
            { cache }
        );
        for (const agent of [this, recipient]) {
            agent.client_cache = agent.previous_cache;
            agent.previous_cache = null;
        }
        const chat_result = new ChatResult({
            chat_history: this.chat_messages[recipient],
            summary,
            cost: gather_usage_summary([this, recipient]),
            human_input: this._human_input
        });
        return chat_result;
    }


    async a_initiate_chat(
        recipient: ConversableAgent,
        clear_history: boolean = true,
        silent: boolean | null = false,
        cache: AbstractCache | null = null,
        max_turns: number | null = null,
        summary_method: string | ((...args: any[]) => any) | null = DEFAULT_SUMMARY_METHOD,
        summary_args: Record<string, any> = {},
        message: string | ((...args: any[]) => any) | null = null,
        ...kwargs: any[]
    ): Promise<ChatResult> {
        /**
         * (async) Initiate a chat with the recipient agent.
         *
         * Reset the consecutive auto reply counter.
         * If `clear_history` is True, the chat history with the recipient agent will be cleared.
         * `a_generate_init_message` is called to generate the initial message for the agent.
         *
         * Args: Please refer to `initiate_chat`.
         *
         * Returns:
         *     ChatResult: an ChatResult object.
         */
        const _chat_info = { ...arguments };
        _chat_info["sender"] = this;
        consolidate_chat_info(_chat_info, { uniform_sender: this });
        for (const agent of [this, recipient]) {
            agent.previous_cache = agent.client_cache;
            agent.client_cache = cache;
        }
        if (typeof max_turns === "number") {
            this._prepare_chat(recipient, clear_history, false);
            for (let i = 0; i < max_turns; i++) {
                let msg2send;
                if (i === 0) {
                    if (typeof message === "function") {
                        msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs);
                    } else {
                        msg2send = await this.a_generate_init_message(message, ...kwargs);
                    }
                } else {
                    msg2send = await this.a_generate_reply({ messages: this.chat_messages[recipient], sender: recipient });
                }
                if (msg2send === null) {
                    break;
                }
                await this.a_send(msg2send, recipient, { request_reply: true, silent });
            }
        } else {
            this._prepare_chat(recipient, clear_history);
            let msg2send;
            if (typeof message === "function") {
                msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs);
            } else {
                msg2send = await this.a_generate_init_message(message, ...kwargs);
            }
            await this.a_send(msg2send, recipient, { silent });
        }
        const summary = this._summarize_chat(
            summary_method,
            summary_args,
            recipient,
            cache
        );
        for (const agent of [this, recipient]) {
            agent.client_cache = agent.previous_cache;
            agent.previous_cache = null;
        }
        const chat_result = new ChatResult(
            this.chat_messages[recipient],
            summary,
            gather_usage_summary([this, recipient]),
            this._human_input
        );
        return chat_result;
    }

    _summarize_chat(
        summary_method: string | ((...args: any[]) => any),
        summary_args: Record<string, any>,
        recipient: Agent | null = null,
        cache: AbstractCache | null = null
    ): string {
        /**
         * Get a chat summary from an agent participating in a chat.
         *
         * Args:
         *     summary_method (str or callable): the summary_method to get the summary.
         *         The callable summary_method should take the recipient and sender agent in a chat as input and return a string of summary. E.g,
         *         ```python
         *
         *         def my_summary_method(
         *             sender: ConversableAgent,
         *             recipient: ConversableAgent,
         *             summary_args: dict,
         *         ):
         *             return recipient.last_message(sender)["content"]
         *         ```
         *     summary_args (dict): a dictionary of arguments to be passed to the summary_method.
         *     recipient: the recipient agent in a chat.
         *     cache: the cache client to be used for this conversation. When provided,
         *         the cache will be used to store and retrieve LLM responses when generating
         *         summaries, which can improve performance and reduce API costs for
         *         repetitive summary requests. The cache is passed to the summary_method
         *         via summary_args['cache'].
         *
         * Returns:
         *     str: a chat summary from the agent.
         */
        let summary = "";
        if (summary_method === null) {
            return summary;
        }
        if (!("cache" in summary_args)) {
            summary_args["cache"] = cache;
        }
        if (summary_method === "reflection_with_llm") {
            summary_method = this._reflection_with_llm_as_summary;
        } else if (summary_method === "last_msg") {
            summary_method = this._last_msg_as_summary;
        }

        if (typeof summary_method === "function") {
            summary = summary_method(this, recipient, summary_args);
        } else {
            throw new Error(
                "If not None, the summary_method must be a string from [`reflection_with_llm`, `last_msg`] or a callable."
            );
        }
        return summary;
    }

    static _last_msg_as_summary(sender: any, recipient: any, summary_args: any): string {
        /**
         * Get a chat summary from the last message of the recipient.
         */
        let summary = "";
        try {
            const content = recipient.last_message(sender)["content"];
            if (typeof content === "string") {
                summary = content.replace("TERMINATE", "");
            } else if (Array.isArray(content)) {
                // Remove the `TERMINATE` word in the content list.
                summary = content
                    .filter((x: any) => typeof x === "object" && "text" in x)
                    .map((x: any) => x["text"].replace("TERMINATE", ""))
                    .join("\n");
            }
        } catch (e) {
            if (e instanceof IndexError || e instanceof AttributeError) {
                warnings.warn(`Cannot extract summary using last_msg: ${e}. Using an empty str as summary.`, UserWarning);
            }
        }
        return summary;
    }

    static _reflection_with_llm_as_summary(sender: any, recipient: any, summary_args: any): string {
        let prompt = summary_args["summary_prompt"];
        prompt = ConversableAgent.DEFAULT_SUMMARY_PROMPT ?? prompt;
        if (typeof prompt !== "string") {
            throw new Error("The summary_prompt must be a string.");
        }
        const msg_list = recipient.chat_messages_for_summary(sender);
        const agent = recipient ?? sender;
        const role = summary_args["summary_role"];
        if (role && typeof role !== "string") {
            throw new Error("The summary_role in summary_arg must be a string.");
        }
        let summary = "";
        try {
            summary = sender._reflection_with_llm(
                prompt,
                msg_list,
                agent,
                summary_args["cache"],
                role
            );
        } catch (e) {
            warnings.warn(
                `Cannot extract summary using reflection_with_llm: ${e}. Using an empty str as summary.`,
                UserWarning
            );
            summary = "";
        }
        return summary;
    }

    _reflection_with_llm(
        prompt: string,
        messages: any[],
        llm_agent: Agent | null = null,
        cache: AbstractCache | null = null,
        role: string | null = null
    ): string {
        /**
         * Get a chat summary using reflection with an llm client based on the conversation history.
         *
         * Args:
         *     prompt (str): The prompt (in this method it is used as system prompt) used to get the summary.
         *     messages (list): The messages generated as part of a chat conversation.
         *     llm_agent: the agent with an llm client.
         *     cache (AbstractCache or None): the cache client to be used for this conversation.
         *     role (str): the role of the message, usually "system" or "user". Default is "system".
         */
        if (!role) {
            role = "system";
        }

        const system_msg = [
            {
                role,
                content: prompt
            }
        ];

        messages = messages.concat(system_msg);
        let llm_client;
        if (llm_agent && llm_agent.client !== null) {
            llm_client = llm_agent.client;
        } else if (this.client !== null) {
            llm_client = this.client;
        } else {
            throw new Error("No OpenAIWrapper client is found.");
        }
        const response = this._generate_oai_reply_from_client({ llm_client, messages, cache });
        return response;
    }

    _check_chat_queue_for_sender(chat_queue: Array<Record<string, any>>): Array<Record<string, any>> {
        /**
         * Check the chat queue and add the "sender" key if it's missing.
         *
         * Args:
         *     chat_queue (List[Dict[str, Any]]): A list of dictionaries containing chat information.
         *
         * Returns:
         *     List[Dict[str, Any]]: A new list of dictionaries with the "sender" key added if it was missing.
         */
        const chat_queue_with_sender = [];
        for (const chat_info of chat_queue) {
            if (!chat_info["sender"]) {
                chat_info["sender"] = this;
            }
            chat_queue_with_sender.push(chat_info);
        }
        return chat_queue_with_sender;
    }

    initiate_chats(chat_queue: Array<Record<string, any>>): Array<ChatResult> {
        /**
         * (Experimental) Initiate chats with multiple agents.
         *
         * Args:
         *     chat_queue (List[Dict]): a list of dictionaries containing the information of the chats.
         *         Each dictionary should contain the input arguments for [`initiate_chat`](#initiate-chat)
         *
         * Returns: a list of ChatResult objects corresponding to the finished chats in the chat_queue.
         */
        const _chat_queue = this._check_chat_queue_for_sender(chat_queue);
        this._finished_chats = initiate_chats(_chat_queue);
        return this._finished_chats;
    }

    async a_initiate_chats(chat_queue: Array<Record<string, any>>): Promise<Record<number, ChatResult>> {
        const _chat_queue = this._check_chat_queue_for_sender(chat_queue);
        this._finished_chats = await a_initiate_chats(_chat_queue);
        return this._finished_chats;
    }

    get_chat_results(chat_index: number | null = null): Array<ChatResult> | ChatResult {
        /**
         * A summary from the finished chats of particular agents.
         */
        if (chat_index !== null) {
            return this._finished_chats[chat_index];
        } else {
            return this._finished_chats;
        }
    }

    reset(): void {
        /**
         * Reset the agent.
         */
        this.clear_history();
        this.reset_consecutive_auto_reply_counter();
        this.stop_reply_at_receive();
        if (this.client !== null) {
            this.client.clear_usage_summary();
        }
        for (const reply_func_tuple of this._reply_func_list) {
            if (reply_func_tuple["reset_config"] !== null) {
                reply_func_tuple["reset_config"](reply_func_tuple["config"]);
            } else {
                reply_func_tuple["config"] = { ...reply_func_tuple["init_config"] };
            }
        }
    }

    stop_reply_at_receive(sender: Agent | null = null): void {
        /**
         * Reset the reply_at_receive of the sender.
         */
        if (sender === null) {
            this.reply_at_receive.clear();
        } else {
            this.reply_at_receive[sender] = false;
        }
    }

    reset_consecutive_auto_reply_counter(sender: Agent | null = null): void {
        /**
         * Reset the consecutive_auto_reply_counter of the sender.
         */
        if (sender === null) {
            this._consecutive_auto_reply_counter.clear();
        } else {
            this._consecutive_auto_reply_counter[sender] = 0;
        }
    }


    clearHistory(recipient?: Agent, nrMessagesToPreserve?: number) {
        /**
         * Clear the chat history of the agent.
         *
         * Args:
         *     recipient: the agent with whom the chat history to clear. If None, clear the chat history with all agents.
         *     nr_messages_to_preserve: the number of newest messages to preserve in the chat history.
         */
        const iostream = IOStream.getDefault();
        if (recipient === undefined) {
            let noMessagesPreserved = 0;
            if (nrMessagesToPreserve) {
                for (const key in this._oaiMessages) {
                    let nrMessagesToPreserveInternal = nrMessagesToPreserve;
                    // if breaking history between function call and function response, save function call message
                    // additionally, otherwise openai will return error
                    const firstMsgToSave = this._oaiMessages[key][this._oaiMessages[key].length - nrMessagesToPreserveInternal];
                    if ("tool_responses" in firstMsgToSave) {
                        nrMessagesToPreserveInternal += 1;
                        // clear_conversable_agent_history.print_preserving_message(iostream.print)
                        noMessagesPreserved += 1;
                    }
                    // Remove messages from history except last `nr_messages_to_preserve` messages.
                    this._oaiMessages[key] = this._oaiMessages[key].slice(-nrMessagesToPreserveInternal);
                }
                iostream.send(
                    new ClearConversableAgentHistoryMessage({ agent: this, noMessagesPreserved })
                );
            } else {
                this._oaiMessages.clear();
            }
        } else {
            this._oaiMessages[recipient].clear();
            // clear_conversable_agent_history.print_warning(iostream.print)
            if (nrMessagesToPreserve) {
                iostream.send(new ClearConversableAgentHistoryWarningMessage({ recipient: this }));
            }
        }
    }

    generateOaiReply(
        messages?: Array<{ [key: string]: any }>,
        sender?: Agent,
        config?: OpenAIWrapper
    ): [boolean, string | { [key: string]: any } | null] {
        /** Generate a reply using autogen.oai. */
        const client = config === undefined ? this.client : config;
        if (client === undefined) {
            return [false, null];
        }
        if (messages === undefined) {
            messages = this._oaiMessages[sender];
        }
        const extractedResponse = this._generateOaiReplyFromClient(
            client, this._oaiSystemMessage.concat(messages), this.clientCache
        );
        return extractedResponse === null ? [false, null] : [true, extractedResponse];
    }

    _generateOaiReplyFromClient(llmClient: any, messages: any, cache: any): string | { [key: string]: any } | null {
        // unroll tool_responses
        const allMessages: any[] = [];
        for (const message of messages) {
            const toolResponses = message["tool_responses"] || [];
            if (toolResponses.length > 0) {
                allMessages.push(...toolResponses);
                // tool role on the parent message means the content is just concatenation of all of the tool_responses
                if (message["role"] !== "tool") {
                    allMessages.push(Object.fromEntries(Object.entries(message).filter(([key]) => key !== "tool_responses")));
                }
            } else {
                allMessages.push(message);
            }
        }

        // TODO: #1143 handle token limit exceeded error
        const response = llmClient.create({
            context: messages[messages.length - 1]["context"] || null,
            messages: allMessages,
            cache: cache,
            agent: this,
        });
        const extractedResponse = llmClient.extractTextOrCompletionObject(response)[0];

        if (extractedResponse === null) {
            console.warn(`Extracted_response from ${response} is None.`);
            return null;
        }
        // ensure function and tool calls will be accepted when sent back to the LLM
        if (typeof extractedResponse !== "string" && "model_dump" in extractedResponse) {
            extractedResponse = extractedResponse.model_dump();
        }
        if (typeof extractedResponse === "object") {
            if (extractedResponse["function_call"]) {
                extractedResponse["function_call"]["name"] = this._normalizeName(
                    extractedResponse["function_call"]["name"]
                );
            }
            for (const toolCall of extractedResponse["tool_calls"] || []) {
                toolCall["function"]["name"] = this._normalizeName(toolCall["function"]["name"]);
                // Remove id and type if they are not present.
                // This is to make the tool call object compatible with Mistral API.
                if (toolCall["id"] === undefined) {
                    delete toolCall["id"];
                }
                if (toolCall["type"] === undefined) {
                    delete toolCall["type"];
                }
            }
        }
        return extractedResponse;
    }

    async aGenerateOaiReply(
        messages?: Array<{ [key: string]: any }>,
        sender?: Agent,
        config?: any
    ): Promise<[boolean, string | { [key: string]: any } | null]> {
        /** Generate a reply using autogen.oai asynchronously. */
        const iostream = IOStream.getDefault();

        const _generateOaiReply = (
            self: this, iostream: IOStream, ...args: any[]
        ): [boolean, string | { [key: string]: any } | null] => {
            IOStream.setDefault(iostream);
            return self.generateOaiReply(...args);
        };

        return await new Promise((resolve) => {
            const executor = () => resolve(
                _generateOaiReply(this, iostream, messages, sender, config)
            );
            setImmediate(executor);
        });
    }

    _generateCodeExecutionReplyUsingExecutor(
        messages?: Array<{ [key: string]: any }>,
        sender?: Agent,
        config?: { [key: string]: any } | false
    ) {
        /** Generate a reply using code executor. */
        const iostream = IOStream.getDefault();

        if (config !== undefined) {
            throw new Error("config is not supported for _generate_code_execution_reply_using_executor.");
        }
        if (this._codeExecutionConfig === false) {
            return [false, null];
        }
        if (messages === undefined) {
            messages = this._oaiMessages[sender];
        }
        const lastNMessages = this._codeExecutionConfig["last_n_messages"] || "auto";

        if (!(typeof lastNMessages === "number" && lastNMessages >= 0) && lastNMessages !== "auto") {
            throw new Error("last_n_messages must be either a non-negative integer, or the string 'auto'.");
        }

        let numMessagesToScan = lastNMessages;
        if (lastNMessages === "auto") {
            // Find when the agent last spoke
            numMessagesToScan = 0;
            for (const message of messages.reverse()) {
                if (!("role" in message) || message["role"] !== "user") {
                    break;
                } else {
                    numMessagesToScan += 1;
                }
            }
        }
        numMessagesToScan = Math.min(messages.length, numMessagesToScan);
        const messagesToScan = messages.slice(-numMessagesToScan);

        // iterate through the last n messages in reverse
        // if code blocks are found, execute the code blocks and return the output
        // if no code blocks are found, continue
        for (const message of messagesToScan.reverse()) {
            if (!message["content"]) {
                continue;
            }
            const codeBlocks = this._codeExecutor.codeExtractor.extractCodeBlocks(message["content"]);
            if (codeBlocks.length === 0) {
                continue;
            }

            iostream.send(new GenerateCodeExecutionReplyMessage({ codeBlocks, sender, recipient: this }));

            // found code blocks, execute code.
            const codeResult = this._codeExecutor.executeCodeBlocks(codeBlocks);
            const exitcode2str = codeResult.exitCode === 0 ? "execution succeeded" : "execution failed";
            return [true, `exitcode: ${codeResult.exitCode} (${exitcode2str})\nCode output: ${codeResult.output}`];
        }

        return [false, null];
    }

    generateCodeExecutionReply(
        messages?: Array<{ [key: string]: any }>,
        sender?: Agent,
        config?: { [key: string]: any } | false
    ) {
        /** Generate a reply using code execution. */
        const codeExecutionConfig = config !== undefined ? config : this._codeExecutionConfig;
        if (codeExecutionConfig === false) {
            return [false, null];
        }
        if (messages === undefined) {
            messages = this._oaiMessages[sender];
        }
        const lastNMessages = codeExecutionConfig["last_n_messages"] || "auto";

        if (!(typeof lastNMessages === "number" && lastNMessages >= 0) && lastNMessages !== "auto") {
            throw new Error("last_n_messages must be either a non-negative integer, or the string 'auto'.");
        }

        let messagesToScan = lastNMessages;
        if (lastNMessages === "auto") {
            // Find when the agent last spoke
            messagesToScan = 0;
            for (let i = 0; i < messages.length; i++) {
                const message = messages[messages.length - (i + 1)];
                if (!("role" in message) || message["role"] !== "user") {
                    break;
                } else {
                    messagesToScan += 1;
                }
            }
        }

        // iterate through the last n messages in reverse
        // if code blocks are found, execute the code blocks and return the output
        // if no code blocks are found, continue
        for (let i = 0; i < Math.min(messages.length, messagesToScan); i++) {
            const message = messages[messages.length - (i + 1)];
            if (!message["content"]) {
                continue;
            }
            const codeBlocks = extractCode(message["content"]);
            if (codeBlocks.length === 1 && codeBlocks[0][0] === UNKNOWN) {
                continue;
            }

            // found code blocks, execute code and push "last_n_messages" back
            const [exitcode, logs] = this.executeCodeBlocks(codeBlocks);
            codeExecutionConfig["last_n_messages"] = lastNMessages;
            const exitcode2str = exitcode === 0 ? "execution succeeded" : "execution failed";
            return [true, `exitcode: ${exitcode} (${exitcode2str})\nCode output: ${logs}`];
        }

        // no code blocks are found, push last_n_messages back and return.
        codeExecutionConfig["last_n_messages"] = lastNMessages;

        return [false, null];
    }

    generateFunctionCallReply(
        messages?: Array<{ [key: string]: any }>,
        sender?: Agent,
        config?: any
    ): [boolean, { [key: string]: any } | null] {
        /**
         * Generate a reply using function call.
         *
         * "function_call" replaced by "tool_calls" as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
         * See https://platform.openai.com/docs/api-reference/chat/create#chat-create-functions
         */
        if (config === undefined) {
            config = this;
        }
        if (messages === undefined) {
            messages = this._oaiMessages[sender];
        }
        const message = messages[messages.length - 1];
        if (message["function_call"]) {
            const callId = message["id"] || null;
            const funcCall = message["function_call"];
            const func = this._functionMap[funcCall["name"] || ""];
            if (func && func.constructor.name === "AsyncFunction") {
                let loop;
                let closeLoop = false;
                try {
                    loop = process.nextTick;
                } catch (error) {
                    loop = setImmediate;
                    closeLoop = true;
                }

                const [, funcReturn] = loop(() => this.aExecuteFunction(funcCall, { callId }));
                if (closeLoop) {
                    clearImmediate(loop);
                }
                return [true, funcReturn];
            } else {
                const [, funcReturn] = this.executeFunction(funcCall, { callId });
                return [true, funcReturn];
            }
        }
        return [false, null];
    }

    async aGenerateFunctionCallReply(
        messages?: Array<{ [key: string]: any }>,
        sender?: Agent,
        config?: any
    ): Promise<[boolean, { [key: string]: any } | null]> {
        /**
         * Generate a reply using async function call.
         *
         * "function_call" replaced by "tool_calls" as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
         * See https://platform.openai.com/docs/api-reference/chat/create#chat-create-functions
         */
        if (config === undefined) {
            config = this;
        }
        if (messages === undefined) {
            messages = this._oaiMessages[sender];
        }
        const message = messages[messages.length - 1];
        if ("function_call" in message) {
            const callId = message["id"] || null;
            const funcCall = message["function_call"];
            const funcName = funcCall["name"] || "";
            const func = this._functionMap[funcName];
            if (func && func.constructor.name === "AsyncFunction") {
                const [, funcReturn] = await this.aExecuteFunction(funcCall, { callId });
                return [true, funcReturn];
            } else {
                const [, funcReturn] = this.executeFunction(funcCall, { callId });
                return [true, funcReturn];
            }
        }

        return [false, null];
    }

    _strForToolResponse(toolResponse: { [key: string]: any }): string {
        return String(toolResponse["content"] || "");
    }


    generate_tool_calls_reply(
        messages: Array<Record<string, any>> | null = null,
        sender: Agent | null = null,
        config: any | null = null
    ): [boolean, Record<string, any> | null] {
        /** Generate a reply using tool call. */
        if (config === null) {
            config = this;
        }
        if (messages === null) {
            messages = this._oai_messages[sender];
        }
        const message = messages[messages.length - 1];
        const tool_returns: Array<Record<string, any>> = [];
        for (const tool_call of message.get("tool_calls", [])) {
            const function_call = tool_call.get("function", {});
            const tool_call_id = tool_call.get("id", null);
            const func = this._function_map.get(function_call.get("name", null), null);
            if (inspect.iscoroutinefunction(func)) {
                let loop;
                let close_loop = false;
                try {
                    // get the running loop if it was already created
                    loop = asyncio.get_running_loop();
                } catch (RuntimeError) {
                    // create a loop if there is no running loop
                    loop = asyncio.new_event_loop();
                    close_loop = true;
                }

                const [, func_return] = loop.run_until_complete(this.a_execute_function(function_call, { call_id: tool_call_id }));
                if (close_loop) {
                    loop.close();
                }
            } else {
                const [, func_return] = this.execute_function(function_call, { call_id: tool_call_id });
            }
            let content = func_return.get("content", "");
            if (content === null) {
                content = "";
            }

            let tool_call_response;
            if (tool_call_id !== null) {
                tool_call_response = {
                    tool_call_id: tool_call_id,
                    role: "tool",
                    content: content,
                };
            } else {
                // Do not include tool_call_id if it is not present.
                // This is to make the tool call object compatible with Mistral API.
                tool_call_response = {
                    role: "tool",
                    content: content,
                };
            }
            tool_returns.push(tool_call_response);
        }
        if (tool_returns.length > 0) {
            return [true, {
                role: "tool",
                tool_responses: tool_returns,
                content: tool_returns.map(tool_return => this._str_for_tool_response(tool_return)).join("\n\n"),
            }];
        }
        return [false, null];
    }

    async _a_execute_tool_call(tool_call: Record<string, any>): Promise<Record<string, any>> {
        const tool_call_id = tool_call["id"];
        const function_call = tool_call.get("function", {});
        const [, func_return] = await this.a_execute_function(function_call, { call_id: tool_call_id });
        return {
            tool_call_id: tool_call_id,
            role: "tool",
            content: func_return.get("content", ""),
        };
    }

    async a_generate_tool_calls_reply(
        messages: Array<Record<string, any>> | null = null,
        sender: Agent | null = null,
        config: any | null = null
    ): Promise<[boolean, Record<string, any> | null]> {
        /** Generate a reply using async function call. */
        if (config === null) {
            config = this;
        }
        if (messages === null) {
            messages = this._oai_messages[sender];
        }
        const message = messages[messages.length - 1];
        const async_tool_calls: Array<Promise<Record<string, any>>> = [];
        for (const tool_call of message.get("tool_calls", [])) {
            async_tool_calls.push(this._a_execute_tool_call(tool_call));
        }
        if (async_tool_calls.length > 0) {
            const tool_returns = await Promise.all(async_tool_calls);
            return [true, {
                role: "tool",
                tool_responses: tool_returns,
                content: tool_returns.map(tool_return => this._str_for_tool_response(tool_return)).join("\n\n"),
            }];
        }

        return [false, null];
    }

    check_termination_and_human_reply(
        messages: Array<Record<string, any>> | null = null,
        sender: Agent | null = null,
        config: any | null = null
    ): [boolean, string | null] {
        /** Check if the conversation should be terminated, and if human reply is provided.

        This method checks for conditions that require the conversation to be terminated, such as reaching
        a maximum number of consecutive auto-replies or encountering a termination message. Additionally,
        it prompts for and processes human input based on the configured human input mode, which can be
        'ALWAYS', 'NEVER', or 'TERMINATE'. The method also manages the consecutive auto-reply counter
        for the conversation and prints relevant messages based on the human input received.

        Args:
            messages: A list of message dictionaries, representing the conversation history.
            sender: The agent object representing the sender of the message.
            config: Configuration object, defaults to the current instance if not provided.

        Returns:
            A tuple containing a boolean indicating if the conversation
            should be terminated, and a human reply which can be a string, a dictionary, or None.
        */
        const iostream = IOStream.get_default();

        if (config === null) {
            config = this;
        }
        if (messages === null) {
            messages = sender ? this._oai_messages[sender] : [];
        }

        let termination_reason = null;

        // if there are no messages, continue the conversation
        if (!messages.length) {
            return [false, null];
        }
        const message = messages[messages.length - 1];

        let reply = "";
        let no_human_input_msg = "";
        const sender_name = sender === null ? "the sender" : sender.name;
        if (this.human_input_mode === "ALWAYS") {
            reply = this.get_human_input(
                `Replying as ${this.name}. Provide feedback to ${sender_name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: `
            );
            no_human_input_msg = !reply ? "NO HUMAN INPUT RECEIVED." : "";
            // if the human input is empty, and the message is a termination message, then we will terminate the conversation
            if (!reply && this._is_termination_msg(message)) {
                termination_reason = `Termination message condition on agent '${this.name}' met`;
            } else if (reply === "exit") {
                termination_reason = "User requested to end the conversation";
            }

            reply = reply || (!this._is_termination_msg(message) ? "exit" : "");
        } else {
            if (this._consecutive_auto_reply_counter[sender] >= this._max_consecutive_auto_reply_dict[sender]) {
                if (this.human_input_mode === "NEVER") {
                    termination_reason = "Maximum number of consecutive auto-replies reached";
                    reply = "exit";
                } else {
                    // self.human_input_mode == "TERMINATE":
                    const terminate = this._is_termination_msg(message);
                    reply = this.get_human_input(
                        terminate
                            ? `Please give feedback to ${sender_name}. Press enter or type 'exit' to stop the conversation: `
                            : `Please give feedback to ${sender_name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: `
                    );
                    no_human_input_msg = !reply ? "NO HUMAN INPUT RECEIVED." : "";
                    // if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    if (reply !== "exit" && terminate) {
                        termination_reason = `Termination message condition on agent '${this.name}' met and no human input provided`;
                    } else if (reply === "exit") {
                        termination_reason = "User requested to end the conversation";
                    }

                    reply = reply || (!terminate ? "exit" : "");
                }
            } else if (this._is_termination_msg(message)) {
                if (this.human_input_mode === "NEVER") {
                    termination_reason = `Termination message condition on agent '${this.name}' met`;
                    reply = "exit";
                } else {
                    // self.human_input_mode == "TERMINATE":
                    reply = this.get_human_input(
                        `Please give feedback to ${sender_name}. Press enter or type 'exit' to stop the conversation: `
                    );
                    no_human_input_msg = !reply ? "NO HUMAN INPUT RECEIVED." : "";

                    // if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    if (!reply || reply === "exit") {
                        termination_reason = `Termination message condition on agent '${this.name}' met and no human input provided`;
                    }

                    reply = reply || "exit";
                }
            }
        }

        // print the no_human_input_msg
        if (no_human_input_msg) {
            iostream.send(
                new TerminationAndHumanReplyNoInputMessage({
                    no_human_input_msg: no_human_input_msg,
                    sender: sender,
                    recipient: this
                })
            );
        }

        // stop the conversation
        if (reply === "exit") {
            // reset the consecutive_auto_reply_counter
            this._consecutive_auto_reply_counter[sender] = 0;

            if (termination_reason) {
                iostream.send(new TerminationMessage({ termination_reason: termination_reason }));
            }

            return [true, null];
        }

        // send the human reply
        if (reply || this._max_consecutive_auto_reply_dict[sender] === 0) {
            // reset the consecutive_auto_reply_counter
            this._consecutive_auto_reply_counter[sender] = 0;
            // User provided a custom response, return function and tool failures indicating user interruption
            const tool_returns: Array<Record<string, any>> = [];
            if (message.get("function_call", false)) {
                tool_returns.push({
                    role: "function",
                    name: message["function_call"].get("name", ""),
                    content: "USER INTERRUPTED",
                });
            }

            if (message.get("tool_calls", false)) {
                tool_returns.push(...message["tool_calls"].map(tool_call => ({
                    role: "tool",
                    tool_call_id: tool_call.get("id", ""),
                    content: "USER INTERRUPTED"
                })));
            }

            const response: Record<string, any> = { role: "user", content: reply };
            if (tool_returns.length > 0) {
                response["tool_responses"] = tool_returns;
            }

            return [true, response];
        }

        // increment the consecutive_auto_reply_counter
        this._consecutive_auto_reply_counter[sender] += 1;
        if (this.human_input_mode !== "NEVER") {
            iostream.send(new UsingAutoReplyMessage({ human_input_mode: this.human_input_mode, sender: sender, recipient: this }));
        }

        return [false, null];
    }


    async a_check_termination_and_human_reply(
        messages: Array<{ [key: string]: any }> | null = null,
        sender: Agent | null = null,
        config: any | null = null
    ): Promise<[boolean, string | null]> {
        /**
         * (async) Check if the conversation should be terminated, and if human reply is provided.
         *
         * This method checks for conditions that require the conversation to be terminated, such as reaching
         * a maximum number of consecutive auto-replies or encountering a termination message. Additionally,
         * it prompts for and processes human input based on the configured human input mode, which can be
         * 'ALWAYS', 'NEVER', or 'TERMINATE'. The method also manages the consecutive auto-reply counter
         * for the conversation and prints relevant messages based on the human input received.
         *
         * Args:
         *     messages (Optional[List[Dict]]): A list of message dictionaries, representing the conversation history.
         *     sender (Optional[Agent]): The agent object representing the sender of the message.
         *     config (Optional[Any]): Configuration object, defaults to the current instance if not provided.
         *
         * Returns:
         *     Tuple[bool, Union[str, Dict, None]]: A tuple containing a boolean indicating if the conversation
         *     should be terminated, and a human reply which can be a string, a dictionary, or None.
         */
        const iostream = IOStream.get_default();

        if (config === null) {
            config = this;
        }
        if (messages === null) {
            messages = sender ? this._oai_messages[sender] : [];
        }

        let termination_reason = null;

        const message = messages.length ? messages[messages.length - 1] : {};
        let reply = "";
        let no_human_input_msg = "";
        const sender_name = sender === null ? "the sender" : sender.name;
        if (this.human_input_mode === "ALWAYS") {
            reply = await this.a_get_human_input(
                `Replying as ${this.name}. Provide feedback to ${sender_name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: `
            );
            no_human_input_msg = !reply ? "NO HUMAN INPUT RECEIVED." : "";
            // if the human input is empty, and the message is a termination message, then we will terminate the conversation
            if (!reply && this._is_termination_msg(message)) {
                termination_reason = `Termination message condition on agent '${this.name}' met`;
            } else if (reply === "exit") {
                termination_reason = "User requested to end the conversation";
            }

            reply = reply || (!this._is_termination_msg(message) ? "exit" : "");
        } else {
            if (this._consecutive_auto_reply_counter[sender] >= this._max_consecutive_auto_reply_dict[sender]) {
                if (this.human_input_mode === "NEVER") {
                    termination_reason = "Maximum number of consecutive auto-replies reached";
                    reply = "exit";
                } else {
                    // this.human_input_mode == "TERMINATE":
                    const terminate = this._is_termination_msg(message);
                    reply = await this.a_get_human_input(
                        terminate
                            ? `Please give feedback to ${sender_name}. Press enter or type 'exit' to stop the conversation: `
                            : `Please give feedback to ${sender_name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: `
                    );
                    no_human_input_msg = !reply ? "NO HUMAN INPUT RECEIVED." : "";
                    // if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    if (reply !== "exit" && terminate) {
                        termination_reason = `Termination message condition on agent '${this.name}' met and no human input provided`;
                    } else if (reply === "exit") {
                        termination_reason = "User requested to end the conversation";
                    }

                    reply = reply || (!terminate ? "exit" : "");
                }
            } else if (this._is_termination_msg(message)) {
                if (this.human_input_mode === "NEVER") {
                    termination_reason = `Termination message condition on agent '${this.name}' met`;
                    reply = "exit";
                } else {
                    // this.human_input_mode == "TERMINATE":
                    reply = await this.a_get_human_input(
                        `Please give feedback to ${sender_name}. Press enter or type 'exit' to stop the conversation: `
                    );
                    no_human_input_msg = !reply ? "NO HUMAN INPUT RECEIVED." : "";

                    // if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    if (!reply || reply === "exit") {
                        termination_reason = `Termination message condition on agent '${this.name}' met and no human input provided`;
                    }

                    reply = reply || "exit";
                }
            }
        }

        // print the no_human_input_msg
        if (no_human_input_msg) {
            iostream.send(
                new TerminationAndHumanReplyNoInputMessage({
                    no_human_input_msg: no_human_input_msg,
                    sender: sender,
                    recipient: this
                })
            );
        }

        // stop the conversation
        if (reply === "exit") {
            // reset the consecutive_auto_reply_counter
            this._consecutive_auto_reply_counter[sender] = 0;

            if (termination_reason) {
                iostream.send(new TerminationMessage({ termination_reason: termination_reason }));
            }

            return [true, null];
        }

        // send the human reply
        if (reply || this._max_consecutive_auto_reply_dict[sender] === 0) {
            // User provided a custom response, return function and tool results indicating user interruption
            // reset the consecutive_auto_reply_counter
            this._consecutive_auto_reply_counter[sender] = 0;
            const tool_returns: Array<{ role: string; name?: string; content: string; tool_call_id?: string }> = [];
            if (message["function_call"]) {
                tool_returns.push({
                    role: "function",
                    name: message["function_call"]["name"] || "",
                    content: "USER INTERRUPTED"
                });
            }

            if (message["tool_calls"]) {
                tool_returns.push(...message["tool_calls"].map((tool_call: any) => ({
                    role: "tool",
                    tool_call_id: tool_call["id"] || "",
                    content: "USER INTERRUPTED"
                })));
            }

            const response: { role: string; content: string; tool_responses?: Array<{ role: string; name?: string; content: string; tool_call_id?: string }> } = {
                role: "user",
                content: reply
            };
            if (tool_returns.length) {
                response["tool_responses"] = tool_returns;
            }

            return [true, response];
        }

        // increment the consecutive_auto_reply_counter
        this._consecutive_auto_reply_counter[sender] += 1;
        if (this.human_input_mode !== "NEVER") {
            iostream.send(new UsingAutoReplyMessage({ human_input_mode: this.human_input_mode, sender: sender, recipient: this }));
        }

        return [false, null];
    }

    generate_reply(
        messages: Array<{ [key: string]: any }> | null = null,
        sender: Agent | null = null,
        ...kwargs: any[]
    ): string | { [key: string]: any } | null {
        /**
         * Reply based on the conversation history and the sender.
         *
         * Either messages or sender must be provided.
         * Register a reply_func with `None` as one trigger for it to be activated when `messages` is non-empty and `sender` is `None`.
         * Use registered auto reply functions to generate replies.
         * By default, the following functions are checked in order:
         * 1. check_termination_and_human_reply
         * 2. generate_function_call_reply (deprecated in favor of tool_calls)
         * 3. generate_tool_calls_reply
         * 4. generate_code_execution_reply
         * 5. generate_oai_reply
         * Every function returns a tuple (final, reply).
         * When a function returns final=False, the next function will be checked.
         * So by default, termination and human reply will be checked first.
         * If not terminating and human reply is skipped, execute function or code and return the result.
         * AI replies are generated only when no code execution is performed.
         *
         * Args:
         *     messages: a list of messages in the conversation history.
         *     sender: sender of an Agent instance.
         *     **kwargs (Any): Additional arguments to customize reply generation. Supported kwargs:
         *         - exclude (List[Callable[..., Any]]): A list of reply functions to exclude from
         *         the reply generation process. Functions in this list will be skipped even if
         *         they would normally be triggered.
         *
         * Returns:
         *     str or dict or None: reply. None if no reply is generated.
         */
        if (messages === null && sender === null) {
            const error_msg = `Either messages=${messages} or sender=${sender} must be provided.`;
            logger.error(error_msg);
            throw new AssertionError(error_msg);
        }

        if (messages === null) {
            messages = this._oai_messages[sender];
        }

        // Call the hookable method that gives registered hooks a chance to update agent state, used for their context variables.
        this.update_agent_state_before_reply(messages);

        // Call the hookable method that gives registered hooks a chance to process the last message.
        // Message modifications do not affect the incoming messages or self._oai_messages.
        messages = this.process_last_received_message(messages);

        // Call the hookable method that gives registered hooks a chance to process all messages.
        // Message modifications do not affect the incoming messages or self._oai_messages.
        messages = this.process_all_messages_before_reply(messages);

        for (const reply_func_tuple of this._reply_func_list) {
            const reply_func = reply_func_tuple["reply_func"];
            if ("exclude" in kwargs && kwargs["exclude"].includes(reply_func)) {
                continue;
            }
            if (inspect.iscoroutinefunction(reply_func)) {
                continue;
            }
            if (this._match_trigger(reply_func_tuple["trigger"], sender)) {
                const [final, reply] = reply_func(this, { messages: messages, sender: sender, config: reply_func_tuple["config"] });
                if (logging_enabled()) {
                    log_event(
                        this,
                        "reply_func_executed",
                        {
                            reply_func_module: reply_func.__module__,
                            reply_func_name: reply_func.__name__,
                            final: final,
                            reply: reply
                        }
                    );
                }
                if (final) {
                    return reply;
                }
            }
        }
        return this._default_auto_reply;
    }

    async a_generate_reply(
        messages: Array<{ [key: string]: any }> | null = null,
        sender: Agent | null = null,
        ...kwargs: any[]
    ): Promise<string | { [key: string]: any } | null> {
        /**
         * (async) Reply based on the conversation history and the sender.
         *
         * Either messages or sender must be provided.
         * Register a reply_func with `None` as one trigger for it to be activated when `messages` is non-empty and `sender` is `None`.
         * Use registered auto reply functions to generate replies.
         * By default, the following functions are checked in order:
         * 1. check_termination_and_human_reply
         * 2. generate_function_call_reply
         * 3. generate_tool_calls_reply
         * 4. generate_code_execution_reply
         * 5. generate_oai_reply
         * Every function returns a tuple (final, reply).
         * When a function returns final=False, the next function will be checked.
         * So by default, termination and human reply will be checked first.
         * If not terminating and human reply is skipped, execute function or code and return the result.
         * AI replies are generated only when no code execution is performed.
         *
         * Args:
         *     messages: a list of messages in the conversation history.
         *     sender: sender of an Agent instance.
         *     **kwargs (Any): Additional arguments to customize reply generation. Supported kwargs:
         *         - exclude (List[Callable[..., Any]]): A list of reply functions to exclude from
         *         the reply generation process. Functions in this list will be skipped even if
         *         they would normally be triggered.
         *
         * Returns:
         *     str or dict or None: reply. None if no reply is generated.
         */
        if (messages === null && sender === null) {
            const error_msg = `Either messages=${messages} or sender=${sender} must be provided.`;
            logger.error(error_msg);
            throw new AssertionError(error_msg);
        }

        if (messages === null) {
            messages = this._oai_messages[sender];
        }

        // Call the hookable method that gives registered hooks a chance to update agent state, used for their context variables.
        this.update_agent_state_before_reply(messages);

        // Call the hookable method that gives registered hooks a chance to process the last message.
        // Message modifications do not affect the incoming messages or self._oai_messages.
        messages = this.process_last_received_message(messages);

        // Call the hookable method that gives registered hooks a chance to process all messages.
        // Message modifications do not affect the incoming messages or self._oai_messages.
        messages = this.process_all_messages_before_reply(messages);

        for (const reply_func_tuple of this._reply_func_list) {
            const reply_func = reply_func_tuple["reply_func"];
            if ("exclude" in kwargs && kwargs["exclude"].includes(reply_func)) {
                continue;
            }

            if (this._match_trigger(reply_func_tuple["trigger"], sender)) {
                let final, reply;
                if (inspect.iscoroutinefunction(reply_func)) {
                    [final, reply] = await reply_func(this, { messages: messages, sender: sender, config: reply_func_tuple["config"] });
                } else {
                    [final, reply] = reply_func(this, { messages: messages, sender: sender, config: reply_func_tuple["config"] });
                }
                if (final) {
                    return reply;
                }
            }
        }
        return this._default_auto_reply;
    }


    _match_trigger(trigger: string | Function | Agent | any[] | null, sender: Agent | null): boolean {
        /** Check if the sender matches the trigger.

        Args:
            trigger (Union[None, str, type, Agent, Callable, List]): The condition to match against the sender.
            Can be `None`, string, type, `Agent` instance, callable, or a list of these.
            sender (Agent): The sender object or type to be matched against the trigger.

        Returns:
            `True` if the sender matches the trigger, otherwise `False`.

        Raises:
            ValueError: If the trigger type is unsupported.
        */
        if (trigger === null) {
            return sender === null;
        } else if (typeof trigger === 'string') {
            if (sender === null) {
                throw new SenderRequiredError();
            }
            return trigger === sender.name;
        } else if (typeof trigger === 'function') {
            return trigger(sender);
        } else if (trigger instanceof Agent) {
            return trigger === sender;
        } else if (Array.isArray(trigger)) {
            return trigger.some(t => this._match_trigger(t, sender));
        } else {
            throw new Error(`Unsupported trigger type: ${typeof trigger}`);
        }
    }

    get_human_input(prompt: string): string {
        /** Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.

        Returns:
            str: human input.
        */
        const iostream = IOStream.get_default();

        const reply = iostream.input(prompt);
        this._human_input.push(reply);
        return reply;
    }

    async a_get_human_input(prompt: string): Promise<string> {
        /** (Async) Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.

        Returns:
            str: human input.
        */
        const loop = asyncio.get_running_loop();
        const reply = await loop.run_in_executor(null, () => this.get_human_input(prompt));
        return reply;
    }

    run_code(code: string, ...kwargs: any[]): [number, string, string | null] {
        /** Run the code and return the result.

        Override this function to modify the way to run the code.

        Args:
            code (str): the code to be executed.
            **kwargs: other keyword arguments.

        Returns:
            A tuple of (exitcode, logs, image).
            exitcode (int): the exit code of the code execution.
            logs (str): the logs of the code execution.
            image (str or None): the docker image used for the code execution.
        */
        return execute_code(code, ...kwargs);
    }

    execute_code_blocks(code_blocks: any[]): [number, string] {
        /** Execute the code blocks and return the result. */
        const iostream = IOStream.get_default();

        let logs_all = "";
        for (let i = 0; i < code_blocks.length; i++) {
            const [lang, code] = code_blocks[i];
            let language = lang;
            if (!language) {
                language = infer_lang(code);
            }

            iostream.send(new ExecuteCodeBlockMessage({ code, language, code_block_count: i, recipient: this }));

            let exitcode, logs, image;
            if (['bash', 'shell', 'sh'].includes(language)) {
                [exitcode, logs, image] = this.run_code(code, { lang: language, ...this._code_execution_config });
            } else if (PYTHON_VARIANTS.includes(language)) {
                const filename = code.startsWith("# filename: ") ? code.slice(11, code.indexOf("\n")).trim() : null;
                [exitcode, logs, image] = this.run_code(code, { lang: "python", filename, ...this._code_execution_config });
            } else {
                [exitcode, logs, image] = [1, `unknown language ${language}`, null];
            }
            if (image !== null) {
                this._code_execution_config["use_docker"] = image;
            }
            logs_all += "\n" + logs;
            if (exitcode !== 0) {
                return [exitcode, logs_all];
            }
        }
        return [0, logs_all];
    }

    static _format_json_str(jstr: string): string {
        /** Remove newlines outside of quotes, and handle JSON escape sequences.

        1. this function removes the newline in the query outside of quotes otherwise json.loads(s) will fail.
            Ex 1:
            "{\n"tool": "python",\n"query": "print('hello')\nprint('world')"\n}" -> "{"tool": "python","query": "print('hello')\nprint('world')"}"
            Ex 2:
            "{\n  \"location\": \"Boston, MA\"\n}" -> "{"location": "Boston, MA"}"

        2. this function also handles JSON escape sequences inside quotes.
            Ex 1:
            '{"args": "a\na\na\ta"}' -> '{"args": "a\\na\\na\\ta"}'
        */
        const result: string[] = [];
        let inside_quotes = false;
        let last_char = " ";
        for (const char of jstr) {
            if (last_char !== "\\" && char === '"') {
                inside_quotes = !inside_quotes;
            }
            last_char = char;
            if (!inside_quotes && char === "\n") {
                continue;
            }
            if (inside_quotes && char === "\n") {
                result.push("\\n");
            } else if (inside_quotes && char === "\t") {
                result.push("\\t");
            } else {
                result.push(char);
            }
        }
        return result.join("");
    }

    execute_function(
        func_call: { [key: string]: any }, call_id: string | null = null, verbose: boolean = false
    ): [boolean, { [key: string]: any }] {
        /** Execute a function call and return the result.

        Override this function to modify the way to execute function and tool calls.

        Args:
            func_call: a dictionary extracted from openai message at "function_call" or "tool_calls" with keys "name" and "arguments".
            call_id: a string to identify the tool call.
            verbose (bool): Whether to send messages about the execution details to the
                output stream. When True, both the function call arguments and the execution
                result will be displayed. Defaults to False.


        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".

        "function_call" deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
        */
        const iostream = IOStream.get_default();

        const func_name = func_call["name"] || "";
        const func = this._function_map[func_name] || null;

        let is_exec_success = false;
        let arguments: any = {};
        let content: string;
        if (func !== null) {
            const input_string = this._format_json_str(func_call["arguments"] || "{}");
            try {
                arguments = JSON.parse(input_string);
            } catch (e) {
                arguments = null;
                content = `Error: ${e}\n The argument must be in JSON format.`;
            }

            if (arguments !== null) {
                iostream.send(new ExecuteFunctionMessage({ func_name, call_id, arguments, recipient: this }));
                try {
                    content = func(...arguments);
                    is_exec_success = true;
                } catch (e) {
                    content = `Error: ${e}`;
                }
            }
        } else {
            content = `Error: Function ${func_name} not found.`;
        }

        if (verbose) {
            iostream.send(new ExecutedFunctionMessage({ func_name, call_id, arguments, content, recipient: this }));
        }

        return [is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": content,
        }];
    }

    async a_execute_function(
        func_call: { [key: string]: any }, call_id: string | null = null, verbose: boolean = false
    ): Promise<[boolean, { [key: string]: any }]> {
        /** Execute an async function call and return the result.

        Override this function to modify the way async functions and tools are executed.

        Args:
            func_call: a dictionary extracted from openai message at key "function_call" or "tool_calls" with keys "name" and "arguments".
            call_id: a string to identify the tool call.
            verbose (bool): Whether to send messages about the execution details to the
                output stream. When True, both the function call arguments and the execution
                result will be displayed. Defaults to False.

        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".

        "function_call" deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
        */
        const iostream = IOStream.get_default();

        const func_name = func_call["name"] || "";
        const func = this._function_map[func_name] || null;

        let is_exec_success = false;
        let arguments: any = {};
        let content: string;
        if (func !== null) {
            const input_string = this._format_json_str(func_call["arguments"] || "{}");
            try {
                arguments = JSON.parse(input_string);
            } catch (e) {
                arguments = null;
                content = `Error: ${e}\n The argument must be in JSON format.`;
            }

            if (arguments !== null) {
                iostream.send(new ExecuteFunctionMessage({ func_name, call_id, arguments, recipient: this }));
                try {
                    if (typeof func === 'function' && func.constructor.name === 'AsyncFunction') {
                        content = await func(...arguments);
                    } else {
                        content = func(...arguments);
                    }
                    is_exec_success = true;
                } catch (e) {
                    content = `Error: ${e}`;
                }
            }
        } else {
            content = `Error: Function ${func_name} not found.`;
        }

        if (verbose) {
            iostream.send(new ExecutedFunctionMessage({ func_name, call_id, arguments, content, recipient: this }));
        }

        return [is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": content,
        }];
    }


    generate_init_message(
        message: string | Record<string, any> | null | undefined, 
        ...kwargs: any[]
    ): string | Record<string, any> {
        /** Generate the initial message for the agent.
        If message is None, input() will be called to get the initial message.

        Args:
            message (str or None): the message to be processed.
            **kwargs: any additional information. It has the following reserved fields:
                "carryover": a string or a list of string to specify the carryover information to be passed to this chat. It can be a string or a list of string.
                    If provided, we will combine this carryover with the "message" content when generating the initial chat
                    message.

        Returns:
            str or dict: the processed message.
        */
        if (message === null || message === undefined) {
            message = this.get_human_input(">");
        }

        return this._handle_carryover(message, kwargs);
    }

    _handle_carryover(
        message: string | Record<string, any>, 
        kwargs: Record<string, any>
    ): string | Record<string, any> {
        if (!kwargs["carryover"]) {
            return message;
        }

        if (typeof message === "string") {
            return this._process_carryover(message, kwargs);
        } else if (typeof message === "object") {
            if (typeof message["content"] === "string") {
                // Makes sure the original message is not mutated
                message = { ...message };
                message["content"] = this._process_carryover(message["content"], kwargs);
            } else if (Array.isArray(message["content"])) {
                // Makes sure the original message is not mutated
                message = { ...message };
                message["content"] = this._process_multimodal_carryover(message["content"], kwargs);
            }
        } else {
            throw new InvalidCarryOverTypeError("Carryover should be a string or a list of strings.");
        }

        return message;
    }

    _process_carryover(content: string, kwargs: Record<string, any>): string {
        // Makes sure there's a carryover
        if (!kwargs["carryover"]) {
            return content;
        }

        // if carryover is string
        if (typeof kwargs["carryover"] === "string") {
            content += "\nContext: \n" + kwargs["carryover"];
        } else if (Array.isArray(kwargs["carryover"])) {
            content += "\nContext: \n" + kwargs["carryover"].map(t => _post_process_carryover_item(t)).join("\n");
        } else {
            throw new InvalidCarryOverTypeError(
                "Carryover should be a string or a list of strings. Not adding carryover to the message."
            );
        }
        return content;
    }

    _process_multimodal_carryover(
        content: Array<Record<string, any>>, 
        kwargs: Record<string, any>
    ): Array<Record<string, any>> {
        /** Prepends the context to a multimodal message. */
        // Makes sure there's a carryover
        if (!kwargs["carryover"]) {
            return content;
        }

        return [{ "type": "text", "text": this._process_carryover("", kwargs) }, ...content];
    }

    async a_generate_init_message(
        message: string | Record<string, any> | null | undefined, 
        ...kwargs: any[]
    ): Promise<string | Record<string, any>> {
        /** Generate the initial message for the agent.
        If message is None, input() will be called to get the initial message.

        Args:
            message (str or None): the message to be processed.
            **kwargs: any additional information. It has the following reserved fields:
                "carryover": a string or a list of string to specify the carryover information to be passed to this chat. It can be a string or a list of string.
                    If provided, we will combine this carryover with the "message" content when generating the initial chat
                    message.

        Returns:
            str or dict: the processed message.
        */
        if (message === null || message === undefined) {
            message = await this.a_get_human_input(">");
        }

        return this._handle_carryover(message, kwargs);
    }

    get tools(): Array<Tool> {
        /** Get the agent's tools (registered for LLM)

        Note this is a copy of the tools list, use add_tool and remove_tool to modify the tools list.
        */
        return [...this._tools];
    }

    remove_tool_for_llm(tool: Tool): void {
        /** Remove a tool (register for LLM tool) */
        try {
            this._register_for_llm({ tool, api_style: "tool", is_remove: true });
            this._tools = this._tools.filter(t => t !== tool);
        } catch (error) {
            if (error instanceof ValueError) {
                throw new ValueError(`Tool ${tool} not found in collection`);
            }
        }
    }

    register_function(
        function_map: Record<string, Callable<any>>, 
        silent_override: boolean = false
    ): void {
        /** Register functions to the agent.

        Args:
            function_map: a dictionary mapping function names to functions. if function_map[name] is None, the function will be removed from the function_map.
            silent_override: whether to print warnings when overriding functions.
        */
        for (const [name, func] of Object.entries(function_map)) {
            this._assert_valid_name(name);
            if (func === null && !(name in this._function_map)) {
                warnings.warn(`The function ${name} to remove doesn't exist`, name);
            }
            if (!silent_override && name in this._function_map) {
                warnings.warn(`Function '${name}' is being overridden.`, UserWarning);
            }
        }
        this._function_map = { ...this._function_map, ...function_map };
        this._function_map = Object.fromEntries(
            Object.entries(this._function_map).filter(([_, v]) => v !== null)
        );
    }

    update_function_signature(
        func_sig: string | Record<string, any>, 
        is_remove: boolean | null, 
        silent_override: boolean = false
    ): void {
        /** Update a function_signature in the LLM configuration for function_call.

        Args:
            func_sig (str or dict): description/name of the function to update/remove to the model. See: https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions
            is_remove: whether removing the function from llm_config with name 'func_sig'
            silent_override: whether to print warnings when overriding functions.

        Deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
        */
        if (!(this.llm_config instanceof Object)) {
            const error_msg = "To update a function signature, agent must have an llm_config";
            logger.error(error_msg);
            throw new AssertionError(error_msg);
        }

        if (is_remove) {
            if (!this.llm_config["functions"] || this.llm_config["functions"].length === 0) {
                const error_msg = `The agent config doesn't have function ${func_sig}.`;
                logger.error(error_msg);
                throw new AssertionError(error_msg);
            } else {
                this.llm_config["functions"] = this.llm_config["functions"].filter(
                    (func: any) => func["name"] !== func_sig
                );
            }
        } else {
            if (typeof func_sig !== "object") {
                throw new ValueError(
                    `The function signature must be of the type dict. Received function signature type ${typeof func_sig}`
                );
            }
            if (!("name" in func_sig)) {
                throw new ValueError(`The function signature must have a 'name' key. Received: ${func_sig}`);
            }
            this._assert_valid_name(func_sig["name"]);
            if ("functions" in this.llm_config) {
                if (!silent_override && this.llm_config["functions"].some(
                    (func: any) => func["name"] === func_sig["name"]
                )) {
                    warnings.warn(`Function '${func_sig['name']}' is being overridden.`, UserWarning);
                }

                this.llm_config["functions"] = [
                    ...this.llm_config["functions"].filter(
                        (func: any) => func["name"] !== func_sig["name"]
                    ),
                    func_sig
                ];
            } else {
                this.llm_config["functions"] = [func_sig];
            }
        }

        // Do this only if llm_config is a dict. If llm_config is LLMConfig, LLMConfig will handle this.
        if (this.llm_config["functions"].length === 0 && typeof this.llm_config === "object") {
            delete this.llm_config["functions"];
        }

        this.client = new OpenAIWrapper({ ...this.llm_config });
    }

    update_tool_signature(
        tool_sig: string | Record<string, any>, 
        is_remove: boolean, 
        silent_override: boolean = false
    ): void {
        /** Update a tool_signature in the LLM configuration for tool_call.

        Args:
            tool_sig (str or dict): description/name of the tool to update/remove to the model. See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
            is_remove: whether removing the tool from llm_config with name 'tool_sig'
            silent_override: whether to print warnings when overriding functions.
        */
        if (!this.llm_config) {
            const error_msg = "To update a tool signature, agent must have an llm_config";
            logger.error(error_msg);
            throw new AssertionError(error_msg);
        }

        if (is_remove) {
            if (!this.llm_config["tools"] || this.llm_config["tools"].length === 0) {
                const error_msg = `The agent config doesn't have tool ${tool_sig}.`;
                logger.error(error_msg);
                throw new AssertionError(error_msg);
            } else {
                const current_tools = this.llm_config["tools"];
                const filtered_tools: any[] = [];

                // Loop through and rebuild tools list without the tool to remove
                for (const tool of current_tools) {
                    const tool_name = tool["function"]["name"];

                    // Match by tool name, or by tool signature
                    const is_different = typeof tool_sig === "string" ? tool_name !== tool_sig : tool !== tool_sig;

                    if (is_different) {
                        filtered_tools.push(tool);
                    }
                }

                this.llm_config["tools"] = filtered_tools;
            }
        } else {
            if (typeof tool_sig !== "object") {
                throw new ValueError(
                    `The tool signature must be of the type dict. Received tool signature type ${typeof tool_sig}`
                );
            }
            this._assert_valid_name(tool_sig["function"]["name"]);
            if ("tools" in this.llm_config && this.llm_config["tools"].length > 0) {
                if (!silent_override && this.llm_config["tools"].some(
                    (tool: any) => tool["function"]["name"] === tool_sig["function"]["name"]
                )) {
                    warnings.warn(`Function '${tool_sig['function']['name']}' is being overridden.`, UserWarning);
                }
                this.llm_config["tools"] = [
                    ...this.llm_config["tools"].filter(
                        (tool: any) => tool["function"]["name"] !== tool_sig["function"]["name"]
                    ),
                    tool_sig
                ];
            } else {
                this.llm_config["tools"] = [tool_sig];
            }
        }

        // Do this only if llm_config is a dict. If llm_config is LLMConfig, LLMConfig will handle this.
        if (this.llm_config["tools"].length === 0 && typeof this.llm_config === "object") {
            delete this.llm_config["tools"];
        }

        this.client = new OpenAIWrapper({ ...this.llm_config });
    }

    can_execute_function(name: string | Array<string>): boolean {
        /** Whether the agent can execute the function. */
        const names = Array.isArray(name) ? name : [name];
        return names.every(n => n in this._function_map);
    }

    get function_map(): Record<string, Callable<any>> {
        /** Return the function map. */
        return this._function_map;
    }

    _wrap_function<F extends Callable<any>>(
        func: F, 
        inject_params: Record<string, any> = {}, 
        { serialize = true }: { serialize?: boolean } = {}
    ): F {
        /** Wrap the function inject chat context parameters and to dump the return value to json.

        Handles both sync and async functions.

        Args:
            func: the function to be wrapped.
            inject_params: the chat context parameters which will be passed to the function.
            serialize: whether to serialize the return value

        Returns:
            The wrapped function.
        */

        @load_basemodels_if_needed
        @functools.wraps(func)

        function _wrapped_func(...args: any[]): any {
            const retval = func(...args, ...inject_params);
            if (logging_enabled()) {
                log_function_use(this, func, args, retval);
            }
            return serialize ? serialize_to_str(retval) : retval;
        }

        @load_basemodels_if_needed
        @functools.wraps(func)

        async function _a_wrapped_func(...args: any[]): Promise<any> {
            const retval = await func(...args, ...inject_params);
            if (logging_enabled()) {
                log_function_use(this, func, args, retval);
            }
            return serialize ? serialize_to_str(retval) : retval;
        }

        const wrapped_func = inspect.isAsyncFunction(func) ? _a_wrapped_func : _wrapped_func;

        // needed for testing
        (wrapped_func as any)._origin = func;

        return wrapped_func as F;
    }

    static

    private _create_tool_if_needed(
        func_or_tool: F | Tool,
        name?: string,
        description?: string
    ): Tool {
        let tool: Tool;
        if (func_or_tool instanceof Tool) {
            tool = func_or_tool;
            // create new tool object if name or description is not None
            if (name || description) {
                tool = new Tool({ func_or_tool: tool, name, description });
            }
        } else if (typeof func_or_tool === 'function') {
            const function_: (...args: any[]) => any = func_or_tool;
            tool = new Tool({ func_or_tool: function_, name, description });
        } else {
            throw new TypeError(`'func_or_tool' must be a function or a Tool object, got '${typeof func_or_tool}' instead.`);
        }
        return tool;
    }

    register_for_llm(
        name?: string,
        description?: string,
        api_style: "function" | "tool" = "tool",
        silent_override: boolean = false
    ): (func_or_tool: F | Tool) => Tool {
        /** Decorator factory for registering a function to be used by an agent.

        It's return value is used to decorate a function to be registered to the agent. The function uses type hints to
        specify the arguments and return type. The function name is used as the default name for the function,
        but a custom name can be provided. The function description is used to describe the function in the
        agent's configuration.

        Args:
            name (optional(str)): name of the function. If None, the function name will be used (default: None).
            description (optional(str)): description of the function (default: None). It is mandatory
                for the initial decorator, but the following ones can omit it.
            api_style: (literal): the API style for function call.
                For Azure OpenAI API, use version 2023-12-01-preview or later.
                `"function"` style will be deprecated. For earlier version use
                `"function"` if `"tool"` doesn't work.
                See [Azure OpenAI documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling?tabs=python) for details.
            silent_override (bool): whether to suppress any override warning messages.

        Returns:
            The decorator for registering a function to be used by an agent.

        Examples:
            ```
            @user_proxy.register_for_execution()
            @agent2.register_for_llm()
            @agent1.register_for_llm(description="This is a very useful function")

            def my_function(a: Annotated[str, "description of a parameter"] = "a", b: int, c=3.14) -> str:
                 return a + str(b * c)
            ```

            For Azure OpenAI versions prior to 2023-12-01-preview, set `api_style`
            to `"function"` if `"tool"` doesn't work:
            ```
            @agent2.register_for_llm(api_style="function")

            def my_function(a: Annotated[str, "description of a parameter"] = "a", b: int, c=3.14) -> str:
                 return a + str(b * c)
            ```

        */

        const _decorator = (
            func_or_tool: F | Tool, name: string = name, description: string = description
        ): Tool => {
            /** Decorator for registering a function to be used by an agent.

            Args:
                func_or_tool: The function or the tool to be registered.
                name: The name of the function or the tool.
                description: The description of the function or the tool.

            Returns:
                The function to be registered, with the _description attribute set to the function description.

            Raises:
                ValueError: if the function description is not provided and not propagated by a previous decorator.
                RuntimeError: if the LLM config is not set up before registering a function.

            */
            const tool = this._create_tool_if_needed(func_or_tool, name, description);

            this._register_for_llm(tool, api_style, silent_override);
            this._tools.push(tool);

            return tool;
        };

        return _decorator;
    }

    private _register_for_llm(
        tool: Tool, api_style: "tool" | "function", is_remove: boolean = false, silent_override: boolean = false
    ): void {
        /**
        Register a tool for LLM.

        Args:
            tool: the tool to be registered.
            api_style: the API style for function call ("tool" or "function").
            is_remove: whether to remove the function or tool.
            silent_override: whether to suppress any override warning messages.

        Returns:
            None
        */
        // register the function to the agent if there is LLM config, raise an exception otherwise
        if (this.llm_config === null) {
            throw new Error("LLM config must be setup before registering a function for LLM.");
        }

        if (api_style === "function") {
            this.update_function_signature(tool.function_schema, is_remove, silent_override);
        } else if (api_style === "tool") {
            this.update_tool_signature(tool.tool_schema, is_remove, silent_override);
        } else {
            throw new Error(`Unsupported API style: ${api_style}`);
        }
    }

    register_for_execution(
        name?: string,
        description?: string,
        serialize: boolean = true,
        silent_override: boolean = false
    ): (func_or_tool: Tool | F) => Tool {
        /** Decorator factory for registering a function to be executed by an agent.

        It's return value is used to decorate a function to be registered to the agent.

        Args:
            name: name of the function. If None, the function name will be used (default: None).
            description: description of the function (default: None).
            serialize: whether to serialize the return value
            silent_override: whether to suppress any override warning messages

        Returns:
            The decorator for registering a function to be used by an agent.

        Examples:
            ```
            @user_proxy.register_for_execution()
            @agent2.register_for_llm()
            @agent1.register_for_llm(description="This is a very useful function")

            def my_function(a: Annotated[str, "description of a parameter"] = "a", b: int, c=3.14):
                 return a + str(b * c)
            ```

        */

        const _decorator = (
            func_or_tool: Tool | F, name: string = name, description: string = description
        ): Tool => {
            /** Decorator for registering a function to be used by an agent.

            Args:
                func_or_tool: the function or the tool to be registered.
                name: the name of the function.
                description: the description of the function.

            Returns:
                The tool to be registered.

            */

            const tool = this._create_tool_if_needed(func_or_tool, name, description);
            const chat_context = new ChatContext(this);
            const chat_context_params = Object.fromEntries(tool._chat_context_param_names.map(param => [param, chat_context]));

            this.register_function(
                { [tool.name]: this._wrap_function(tool.func, chat_context_params, serialize) },
                silent_override
            );

            return tool;
        };

        return _decorator;
    }

    register_model_client(model_client_cls: ModelClient, ...kwargs: any[]): void {
        /** Register a model client.

        Args:
            model_client_cls: A custom client class that follows the Client interface
            **kwargs: The kwargs for the custom client class to be initialized with
        */
        this.client.register_model_client(model_client_cls, ...kwargs);
    }

    register_hook(hookable_method: string, hook: (...args: any[]) => any): void {
        /** Registers a hook to be called by a hookable method, in order to add a capability to the agent.
        Registered hooks are kept in lists (one per hookable method), and are called in their order of registration.

        Args:
            hookable_method: A hookable method name implemented by ConversableAgent.
            hook: A method implemented by a subclass of AgentCapability.
        */
        if (!(hookable_method in this.hook_lists)) {
            throw new Error(`${hookable_method} is not a hookable method.`);
        }
        const hook_list = this.hook_lists[hookable_method];
        if (hook_list.includes(hook)) {
            throw new Error(`${hook} is already registered as a hook.`);
        }
        hook_list.push(hook);
    }

    update_agent_state_before_reply(messages: Array<{ [key: string]: any }>): void {
        /** Calls any registered capability hooks to update the agent's state.
        Primarily used to update context variables.
        Will, potentially, modify the messages.
        */
        const hook_list = this.hook_lists["update_agent_state"];

        // Call each hook (in order of registration) to process the messages.
        for (const hook of hook_list) {
            hook(this, messages);
        }
    }

    process_all_messages_before_reply(messages: Array<{ [key: string]: any }>): Array<{ [key: string]: any }> {
        /** Calls any registered capability hooks to process all messages, potentially modifying the messages. */
        const hook_list = this.hook_lists["process_all_messages_before_reply"];
        // If no hooks are registered, or if there are no messages to process, return the original message list.
        if (hook_list.length === 0 || messages === null) {
            return messages;
        }

        // Call each hook (in order of registration) to process the messages.
        let processed_messages = messages;
        for (const hook of hook_list) {
            processed_messages = hook(processed_messages);
        }
        return processed_messages;
    }

    process_last_received_message(messages: Array<{ [key: string]: any }>): Array<{ [key: string]: any }> {
        /** Calls any registered capability hooks to use and potentially modify the text of the last message,
        as long as the last message is not a function call or exit command.
        */
        // If any required condition is not met, return the original message list.
        const hook_list = this.hook_lists["process_last_received_message"];
        if (hook_list.length === 0) {
            return messages; // No hooks registered.
        }
        if (messages === null) {
            return null; // No message to process.
        }
        if (messages.length === 0) {
            return messages; // No message to process.
        }
        const last_message = messages[messages.length - 1];
        if ("function_call" in last_message) {
            return messages; // Last message is a function call.
        }
        if ("context" in last_message) {
            return messages; // Last message contains a context key.
        }
        if (!("content" in last_message)) {
            return messages; // Last message has no content.
        }

        const user_content = last_message["content"];
        if (typeof user_content !== 'string' && !Array.isArray(user_content)) {
            // if the user_content is a string, it is for regular LLM
            // if the user_content is a list, it should follow the multimodal LMM format.
            return messages;
        }
        if (user_content === "exit") {
            return messages; // Last message is an exit command.
        }

        // Call each hook (in order of registration) to process the user's message.
        let processed_user_content = user_content;
        for (const hook of hook_list) {
            processed_user_content = hook(processed_user_content);
        }

        if (processed_user_content === user_content) {
            return messages; // No hooks actually modified the user's message.
        }

        // Replace the last user message with the expanded one.
        messages = [...messages];
        messages[messages.length - 1]["content"] = processed_user_content;
        return messages;
    }

    print_usage_summary(mode: string | string[] = ["actual", "total"]): void {
        /** Print the usage summary. */
        const iostream = IOStream.get_default();
        if (this.client === null) {
            iostream.send(new ConversableAgentUsageSummaryNoCostIncurredMessage({ recipient: this }));
        } else {
            iostream.send(new ConversableAgentUsageSummaryMessage({ recipient: this }));
        }

        if (this.client !== null) {
            this.client.print_usage_summary(mode);
        }
    }

    get_actual_usage(): null | { [key: string]: number } {
        /** Get the actual usage summary. */
        if (this.client === null) {
            return null;
        } else {
            return this.client.actual_usage_summary;
        }
    }


    get_total_usage(): null | Record<string, number> {
        /** Get the total usage summary. */
        if (this.client === null) {
            return null;
        } else {
            return this.client.total_usage_summary;
        }
    }

    _create_or_get_executor(
        executor_kwargs: Record<string, any> | null = null,
        tools: Tool | Iterable<Tool> | null = null,
        agent_name: string = "executor",
        agent_human_input_mode: string = "NEVER",
    ): Generator<ConversableAgent, void, void> {
        /** Creates a user proxy / tool executor agent.

        Note: Code execution is not enabled by default. Pass the code execution config into executor_kwargs, if needed.

        Args:
            executor_kwargs: agent's arguments.
            tools: tools to register for execution with the agent.
            agent_name: agent's name, defaults to 'executor'.
            agent_human_input_mode: agent's human input mode, defaults to 'NEVER'.
        */
        if (executor_kwargs === null) {
            executor_kwargs = {};
        }
        if (!("is_termination_msg" in executor_kwargs)) {
            executor_kwargs["is_termination_msg"] = (x: any) => (x["content"] !== null) && x["content"].includes("TERMINATE");
        }

        try {
            if (!this.run_executor) {
                this.run_executor = new ConversableAgent({
                    name: agent_name,
                    human_input_mode: agent_human_input_mode,
                    ...executor_kwargs,
                });
            }

            tools = tools === null ? [] : tools;
            tools = tools instanceof Tool ? [tools] : tools;
            for (const tool of tools) {
                tool.register_for_execution(this.run_executor);
                tool.register_for_llm(this);
            }
            yield this.run_executor;
        } finally {
            if (tools !== null) {
                for (const tool of tools) {
                    this.update_tool_signature({ tool_sig: tool.tool_schema["function"]["name"], is_remove: true });
                }
            }
        }
    }

    run(
        message: string,
        tools: Tool | Iterable<Tool> | null = null,
        executor_kwargs: Record<string, any> | null = null,
        max_turns: number | null = null,
        msg_to: "agent" | "user" = "agent",
        clear_history: boolean = false,
        user_input: boolean = true,
        summary_method: string | ((...args: any[]) => any) | null = DEFAULT_SUMMARY_METHOD,
    ): ChatResult {
        /** Run a chat with the agent using the given message.

        A second agent will be created to represent the user, this agent will by known by the name 'user'. This agent does not have code execution enabled by default, if needed pass the code execution config in with the executor_kwargs parameter.

        The user can terminate the conversation when prompted or, if agent's reply contains 'TERMINATE', it will terminate.

        Args:
            message: the message to be processed.
            tools: the tools to be used by the agent.
            executor_kwargs: the keyword arguments for the executor.
            max_turns: maximum number of turns (a turn is equivalent to both agents having replied), defaults no None which means unlimited. The original message is included.
            msg_to: which agent is receiving the message and will be the first to reply, defaults to the agent.
            clear_history: whether to clear the chat history.
            user_input: the user will be asked for input at their turn.
            summary_method: the method to summarize the chat.
        */
        with (this._create_or_get_executor({
            executor_kwargs,
            tools,
            agent_name: "user",
            agent_human_input_mode: user_input ? "ALWAYS" : "NEVER",
        })) {
            if (msg_to === "agent") {
                return executor.initiate_chat({
                    agent: this,
                    message,
                    clear_history,
                    max_turns,
                    summary_method,
                });
            } else {
                return this.initiate_chat({
                    agent: executor,
                    message,
                    clear_history,
                    max_turns,
                    summary_method,
                });
            }
        }
    }

    async a_run(
        message: string,
        tools: Tool | Iterable<Tool> | null = null,
        executor_kwargs: Record<string, any> | null = null,
        max_turns: number | null = null,
        msg_to: "agent" | "user" = "agent",
        clear_history: boolean = false,
        user_input: boolean = true,
        summary_method: string | ((...args: any[]) => any) | null = DEFAULT_SUMMARY_METHOD,
    ): Promise<ChatResult> {
        /** Run a chat asynchronously with the agent using the given message.

        A second agent will be created to represent the user, this agent will by known by the name 'user'.

        The user can terminate the conversation when prompted or, if agent's reply contains 'TERMINATE', it will terminate.

        Args:
            message: the message to be processed.
            tools: the tools to be used by the agent.
            executor_kwargs: the keyword arguments for the executor.
            max_turns: maximum number of turns (a turn is equivalent to both agents having replied), defaults no None which means unlimited. The original message is included.
            msg_to: which agent is receiving the message and will be the first to reply, defaults to the agent.
            clear_history: whether to clear the chat history.
            user_input: the user will be asked for input at their turn.
            summary_method: the method to summarize the chat.
        */
        with (this._create_or_get_executor({
            executor_kwargs,
            tools,
            agent_name: "user",
            agent_human_input_mode: user_input ? "ALWAYS" : "NEVER",
        })) {
            if (msg_to === "agent") {
                return await executor.a_initiate_chat({
                    agent: this,
                    message,
                    clear_history,
                    max_turns,
                    summary_method,
                });
            } else {
                return await this.a_initiate_chat({
                    agent: executor,
                    message,
                    clear_history,
                    max_turns,
                    summary_method,
                });
            }
        }
    }
}



export function register_function(
  f: (...args: any[]) => any,
  caller: ConversableAgent,
  executor: ConversableAgent,
  name: string | null = null,
  description: string
): void {
  /** Register a function to be proposed by an agent and executed for an executor.

  This function can be used instead of function decorators `@ConversationAgent.register_for_llm` and
  `@ConversationAgent.register_for_execution`.

  Args:
      f: the function to be registered.
      caller: the agent calling the function, typically an instance of ConversableAgent.
      executor: the agent executing the function, typically an instance of UserProxy.
      name: name of the function. If None, the function name will be used (default: None).
      description: description of the function. The description is used by LLM to decode whether the function
          is called. Make sure the description is properly describing what the function does or it might not be
          called by LLM when needed.

  */
  f = caller.register_for_llm({ name, description })(f);
  executor.register_for_execution({ name })(f);
}