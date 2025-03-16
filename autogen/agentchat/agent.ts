// Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
// SPDX-License-Identifier: MIT



export abstract class Agent {
    /**
     * (In preview) A protocol for Agent.
     *
     * An agent can communicate with other agents and perform actions.
     * Different agents can differ in what actions they perform in the `receive` method.
     */

    abstract get name(): string;
    /**
     * The name of the agent.
     */

    abstract get description(): string;
    /**
     * The description of the agent. Used for the agent's introduction in
     * a group chat setting.
     */

    abstract send(
        message: Record<string, any> | string,
        recipient: Agent,
        request_reply?: boolean
    ): void;
    /**
     * Send a message to another agent.
     *
     * Args:
     *     message (dict or str): the message to send. If a dict, it should be
     *         a JSON-serializable and follows the OpenAI's ChatCompletion schema.
     *     recipient (Agent): the recipient of the message.
     *     request_reply (bool): whether to request a reply from the recipient.
     */

    abstract a_send(
        message: Record<string, any> | string,
        recipient: Agent,
        request_reply?: boolean
    ): Promise<void>;
    /**
     * (Async) Send a message to another agent.
     *
     * Args:
     *     message (dict or str): the message to send. If a dict, it should be
     *         a JSON-serializable and follows the OpenAI's ChatCompletion schema.
     *     recipient (Agent): the recipient of the message.
     *     request_reply (bool): whether to request a reply from the recipient.
     */

    abstract receive(
        message: Record<string, any> | string,
        sender: Agent,
        request_reply?: boolean
    ): void;
    /**
     * Receive a message from another agent.
     *
     * Args:
     *     message (dict or str): the message received. If a dict, it should be
     *         a JSON-serializable and follows the OpenAI's ChatCompletion schema.
     *     sender (Agent): the sender of the message.
     *     request_reply (bool): whether the sender requests a reply.
     */

    abstract a_receive(
        message: Record<string, any> | string,
        sender: Agent,
        request_reply?: boolean
    ): Promise<void>;
    /**
     * (Async) Receive a message from another agent.
     *
     * Args:
     *     message (dict or str): the message received. If a dict, it should be
     *         a JSON-serializable and follows the OpenAI's ChatCompletion schema.
     *     sender (Agent): the sender of the message.
     *     request_reply (bool): whether the sender requests a reply.
     */

    abstract generate_reply(
        messages?: Array<Record<string, any>>,
        sender?: Agent,
        ...kwargs: any[]
    ): string | Record<string, any> | null;
    /**
     * Generate a reply based on the received messages.
     *
     * Args:
     *     messages (list[dict[str, Any]]): a list of messages received from other agents.
     *         The messages are dictionaries that are JSON-serializable and
     *         follows the OpenAI's ChatCompletion schema.
     *     sender: sender of an Agent instance.
     *     **kwargs: Additional keyword arguments.
     *
     * Returns:
     *     str or dict or None: the generated reply. If None, no reply is generated.
     */

    abstract a_generate_reply(
        messages?: Array<Record<string, any>>,
        sender?: Agent,
        ...kwargs: any[]
    ): Promise<string | Record<string, any> | null>;
    /**
     * (Async) Generate a reply based on the received messages.
     *
     * Args:
     *     messages (list[dict[str, Any]]): a list of messages received from other agents.
     *         The messages are dictionaries that are JSON-serializable and
     *         follows the OpenAI's ChatCompletion schema.
     *     sender: sender of an Agent instance.
     *     **kwargs: Additional keyword arguments.
     *
     * Returns:
     *     str or dict or None: the generated reply. If None, no reply is generated.
     */
}

export abstract class LLMAgent extends Agent {
    /** (In preview) A protocol for an LLM agent. */

    abstract get system_message(): string;
    /** The system message of this agent. */

    abstract update_system_message(system_message: string): void;
    /** Update this agent's system message.

        Args:
            system_message (str): system message for inference.
        */
}

// if TYPE_CHECKING:
//     # mypy will fail if Conversible agent does not implement Agent protocol

//     function _check_protocol_implementation(agent: ConversableAgent): Agent {
//         return agent;
//     }