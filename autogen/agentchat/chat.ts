// Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
// SPDX-License-Identifier: MIT

import * as asyncio from "asyncio";
import * as datetime from "datetime";
import * as logging from "logging";
import * as warnings from "warnings";
import { defaultdict } from "collections";
import { dataclass } from "dataclasses";
import { partial } from "functools";
import { IOStream } from "..io.base.js";
import { PostCarryoverProcessingMessage } from "..messages.agent_messages.js";
import { consolidate_chat_info } from ".utils.js";

const logger = logging.getLogger(__name__);
const Prerequisite: [number, number];


export type ChatResult = {
    /** (Experimental) The result of a chat. Almost certain to be changed. */
    chat_id: number;
    /** chat id */
    chat_history: Array<{[key: string]: any}>;
    /** The chat history. */
    summary: string;
    /** A summary obtained from the chat. */
    cost: {[key: string]: {[key: string]: any}};
    /** The cost of the chat.
       The value for each usage type is a dictionary containing cost information for that specific type.
           - "usage_including_cached_inference": Cost information on the total usage, including the tokens in cached inference.
           - "usage_excluding_cached_inference": Cost information on the usage of tokens, excluding the tokens in cache. No larger than "usage_including_cached_inference".
    */
    human_input: Array<string>;
    /** A list of human input solicited during the chat. */
};

function _validate_recipients(chat_queue: Array<{[key: string]: any}>): void {
    /** Validate recipients exits and warn repetitive recipients. */
    const receipts_set = new Set();
    for (const chat_info of chat_queue) {
        if (!("recipient" in chat_info)) {
            throw new Error("recipient must be provided.");
        }
        receipts_set.add(chat_info["recipient"]);
    }
    if (receipts_set.size < chat_queue.length) {
        console.warn(
            "Repetitive recipients detected: The chat history will be cleared by default if a recipient appears more than once. To retain the chat history, please set 'clear_history=False' in the configuration of the repeating agent.",
            "UserWarning"
        );
    }
}

function __create_async_prerequisites(chat_queue: Array<{[key: string]: any}>): Array<[number, number]> {
    /** Create list of Prerequisite (prerequisite_chat_id, chat_id) */
    const prerequisites: Array<[number, number]> = [];
    for (const chat_info of chat_queue) {
        if (!("chat_id" in chat_info)) {
            throw new Error("Each chat must have a unique id for async multi-chat execution.");
        }
        const chat_id = chat_info["chat_id"];
        const pre_chats = chat_info["prerequisites"] || [];
        for (const pre_chat_id of pre_chats) {
            if (typeof pre_chat_id !== "number") {
                throw new Error("Prerequisite chat id is not int.");
            }
            prerequisites.push([chat_id, pre_chat_id]);
        }
    }
    return prerequisites;
}

function __find_async_chat_order(chat_ids: Set<number>, prerequisites: Array<[number, number]>): Array<number> {
    /** Find chat order for async execution based on the prerequisite chats

    Args:
        chat_ids: A set of all chat IDs that need to be scheduled
        prerequisites: A list of tuples (chat_id, prerequisite_chat_id) where each tuple indicates that chat_id depends on prerequisite_chat_id

    Returns:
        list: a list of chat_id in order.
    */
    const edges = new Map<number, Set<number>>();
    const indegree = new Map<number, number>();
    for (const pair of prerequisites) {
        const [chat, pre] = pair;
        if (!edges.has(pre)) {
            edges.set(pre, new Set());
        }
        if (!edges.get(pre).has(chat)) {
            indegree.set(chat, (indegree.get(chat) || 0) + 1);
            edges.get(pre).add(chat);
        }
    }
    let bfs = Array.from(chat_ids).filter(i => !indegree.has(i));
    const chat_order: Array<number> = [];
    const steps = indegree.size;
    for (let _ = 0; _ < steps + 1; _++) {
        if (!bfs.length) {
            break;
        }
        chat_order.push(...bfs);
        const nxt: Array<number> = [];
        for (const node of bfs) {
            if (edges.has(node)) {
                for (const course of edges.get(node)) {
                    indegree.set(course, indegree.get(course) - 1);
                    if (indegree.get(course) === 0) {
                        nxt.push(course);
                        indegree.delete(course);
                    }
                }
                edges.delete(node);
            }
        }
        bfs = nxt;
    }

    if (indegree.size) {
        return [];
    }
    return chat_order;
}

function _post_process_carryover_item(carryover_item: any): string {
    if (typeof carryover_item === 'string') {
        return carryover_item;
    } else if (typeof carryover_item === 'object' && 'content' in carryover_item) {
        return String(carryover_item['content']);
    } else {
        return String(carryover_item);
    }
}

function __post_carryover_processing(chat_info: Record<string, any>): void {
    const iostream = IOStream.get_default();

    if (!('message' in chat_info)) {
        warnings.warn(
            "message is not provided in a chat_queue entry. input() will be called to get the initial message.",
            UserWarning
        );
    }

    iostream.send(new PostCarryoverProcessingMessage({ chat_info }));
}

function initiate_chats(chat_queue: Array<Record<string, any>>): Array<ChatResult> {
    /** Initiate a list of chats.

    Args:
        chat_queue (List[Dict]): A list of dictionaries containing the information about the chats.

            Each dictionary should contain the input arguments for
            [`ConversableAgent.initiate_chat`](/docs/api-reference/autogen/ConversableAgent#initiate-chat).
            For example:
                - `"sender"` - the sender agent.
                - `"recipient"` - the recipient agent.
                - `"clear_history"` (bool) - whether to clear the chat history with the agent.
                Default is True.
                - `"silent"` (bool or None) - (Experimental) whether to print the messages in this
                conversation. Default is False.
                - `"cache"` (Cache or None) - the cache client to use for this conversation.
                Default is None.
                - `"max_turns"` (int or None) - maximum number of turns for the chat. If None, the chat
                will continue until a termination condition is met. Default is None.
                - `"summary_method"` (str or callable) - a string or callable specifying the method to get
                a summary from the chat. Default is DEFAULT_summary_method, i.e., "last_msg".
                - `"summary_args"` (dict) - a dictionary of arguments to be passed to the summary_method.
                Default is {}.
                - `"message"` (str, callable or None) - if None, input() will be called to get the
                initial message.
                - `**context` - additional context information to be passed to the chat.
                - `"carryover"` - It can be used to specify the carryover information to be passed
                to this chat. If provided, we will combine this carryover with the "message" content when
                generating the initial chat message in `generate_init_message`.
                - `"finished_chat_indexes_to_exclude_from_carryover"` - It can be used by specifying a list of indexes of the finished_chats list,

    Returns:
        (list): a list of ChatResult objects corresponding to the finished chats in the chat_queue.
    */
    consolidate_chat_info(chat_queue);
    _validate_recipients(chat_queue);
    const current_chat_queue = [...chat_queue];
    const finished_chats: ChatResult[] = [];
    while (current_chat_queue.length) {
        const chat_info = current_chat_queue.shift();
        let _chat_carryover = chat_info?.carryover || [];
        const finished_chat_indexes_to_exclude_from_carryover = chat_info?.finished_chat_indexes_to_exclude_from_carryover || [];

        if (typeof _chat_carryover === 'string') {
            _chat_carryover = [_chat_carryover];
        }
        chat_info.carryover = _chat_carryover.concat(
            finished_chats.filter((_, i) => !finished_chat_indexes_to_exclude_from_carryover.includes(i)).map(r => r.summary)
        );

        if (!chat_info.silent) {
            __post_carryover_processing(chat_info);
        }

        const sender = chat_info.sender;
        const chat_res = sender.initiate_chat(chat_info);
        finished_chats.push(chat_res);
    }
    return finished_chats;
}

function __system_now_str(): string {
    const ct = new Date();
    return ` System time at ${ct}. `;
}

function _on_chat_future_done(chat_future: Promise<any>, chat_id: number): void {
    /** Update ChatResult when async Task for Chat is completed. */
    logger.debug(`Update chat ${chat_id} result on task completion.` + __system_now_str());
    chat_future.then(chat_result => {
        chat_result.chat_id = chat_id;
    });
}


export async function _dependent_chat_future(
  chat_id: number,
  chat_info: Record<string, any>,
  prerequisite_chat_futures: Record<number, Promise<any>>
): Promise<Task<any>> {
  /** Create an async Task for each chat. */
  logger.debug(`Create Task for chat ${chat_id}.` + __system_now_str());
  let _chat_carryover = chat_info.get("carryover", []);
  const finished_chat_indexes_to_exclude_from_carryover = chat_info.get(
    "finished_chat_indexes_to_exclude_from_carryover",
    []
  );
  const finished_chats: Record<number, any> = {};
  for (const chat in prerequisite_chat_futures) {
    const chat_future = prerequisite_chat_futures[chat];
    if (chat_future.cancelled()) {
      throw new Error(`Chat ${chat} is cancelled.`);
    }

    // wait for prerequisite chat results for the new chat carryover
    finished_chats[chat] = await chat_future;
  }

  if (typeof _chat_carryover === "string") {
    _chat_carryover = [_chat_carryover];
  }
  const data = Object.entries(finished_chats)
    .filter(([chat_id]) =>
      !finished_chat_indexes_to_exclude_from_carryover.includes(chat_id)
    )
    .map(([, chat_result]) => chat_result.summary);
  chat_info["carryover"] = _chat_carryover.concat(data);
  if (!chat_info.get("silent", false)) {
    __post_carryover_processing(chat_info);
  }

  const sender = chat_info["sender"];
  const chat_res_future = sender.a_initiate_chat(chat_info);
  const call_back_with_args = _on_chat_future_done.bind(null, { chat_id });
  chat_res_future.then(call_back_with_args);
  logger.debug(`Task for chat ${chat_id} created.` + __system_now_str());
  return chat_res_future;
}

export async function a_initiate_chats(chat_queue: Array<Record<string, any>>): Promise<Record<number, ChatResult>> {
  /** (async) Initiate a list of chats.

  Args:
      chat_queue (List[Dict]): A list of dictionaries containing the information about the chats.

          Each dictionary should contain the input arguments for
          [`ConversableAgent.initiate_chat`](/docs/api-reference/autogen/ConversableAgent#initiate-chat).
          For example:
              - `"sender"` - the sender agent.
              - `"recipient"` - the recipient agent.
              - `"clear_history"` (bool) - whether to clear the chat history with the agent.
              Default is True.
              - `"silent"` (bool or None) - (Experimental) whether to print the messages in this
              conversation. Default is False.
              - `"cache"` (Cache or None) - the cache client to use for this conversation.
              Default is None.
              - `"max_turns"` (int or None) - maximum number of turns for the chat. If None, the chat
              will continue until a termination condition is met. Default is None.
              - `"summary_method"` (str or callable) - a string or callable specifying the method to get
              a summary from the chat. Default is DEFAULT_summary_method, i.e., "last_msg".
              - `"summary_args"` (dict) - a dictionary of arguments to be passed to the summary_method.
              Default is {}.
              - `"message"` (str, callable or None) - if None, input() will be called to get the
              initial message.
              - `**context` - additional context information to be passed to the chat.
              - `"carryover"` - It can be used to specify the carryover information to be passed
              to this chat. If provided, we will combine this carryover with the "message" content when
              generating the initial chat message in `generate_init_message`.
              - `"finished_chat_indexes_to_exclude_from_carryover"` - It can be used by specifying a list of indexes of the finished_chats list,


  Returns:
      - (Dict): a dict of ChatId: ChatResult corresponding to the finished chats in the chat_queue.
  */
  consolidate_chat_info(chat_queue);
  _validate_recipients(chat_queue);
  const chat_book = chat_queue.reduce((acc, chat_info) => {
    acc[chat_info["chat_id"]] = chat_info;
    return acc;
  }, {} as Record<number, any>);
  const num_chats = Object.keys(chat_book);
  const prerequisites = __create_async_prerequisites(chat_queue);
  const chat_order_by_id = __find_async_chat_order(num_chats, prerequisites);
  const finished_chat_futures: Record<number, Promise<any>> = {};
  for (const chat_id of chat_order_by_id) {
    const chat_info = chat_book[chat_id];
    const prerequisite_chat_ids = chat_info.get("prerequisites", []);
    const pre_chat_futures: Record<number, Promise<any>> = {};
    for (const pre_chat_id of prerequisite_chat_ids) {
      const pre_chat_future = finished_chat_futures[pre_chat_id];
      pre_chat_futures[pre_chat_id] = pre_chat_future;
    }
    const current_chat_future = await _dependent_chat_future(chat_id, chat_info, pre_chat_futures);
    finished_chat_futures[chat_id] = current_chat_future;
  }
  await Promise.all(Object.values(finished_chat_futures));
  const finished_chats: Record<number, ChatResult> = {};
  for (const chat in finished_chat_futures) {
    const chat_result = finished_chat_futures[chat].result();
    finished_chats[chat] = chat_result;
  }
  return finished_chats;
}