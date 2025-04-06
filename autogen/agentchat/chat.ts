// Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
// SPDX-License-Identifier: MIT

import { IOStream } from "../io/base";
import { post_carryover_processing_message } from "../messages/agent_messages";
import { consolidate_chat_info } from "./utils";

// Define logger
const logger = console; // TypeScript equivalent of Python's logging.getLogger

// Define Prerequisite type
type Prerequisite = [number, number];

/**
 * (Experimental) The result of a chat. Almost certain to be changed.
 */
export class ChatResult {
  /**
   * chat id
   */
  chatId: number | null = null;

  /**
   * The chat history.
   */
  chatHistory: Array<Record<string, any>> | null = null;

  /**
   * A summary obtained from the chat.
   */
  summary: string | null = null;

  /**
   * The cost of the chat.
   * The value for each usage type is a dictionary containing cost information for that specific type.
   *     - "usage_including_cached_inference": Cost information on the total usage, including the tokens in cached inference.
   *     - "usage_excluding_cached_inference": Cost information on the usage of tokens, excluding the tokens in cache. No larger than "usage_including_cached_inference".
   */
  cost: Record<string, Record<string, any>> | null = null; // keys: "usage_including_cached_inference", "usage_excluding_cached_inference"

  /**
   * A list of human input solicited during the chat.
   */
  humanInput: Array<string> | null = null;
}

/**
 * Validate recipients exits and warn repetitive recipients.
 */
function validate_recipients(chatQueue: Array<Record<string, any>>): void {
  const receiptsSet = new Set();
  for (const chatInfo of chatQueue) {
    if (!("recipient" in chatInfo)) throw new Error("recipient must be provided.");
    receiptsSet.add(chatInfo.recipient);
  }
  if (receiptsSet.size < chatQueue.length) {
    console.warn(
      "Repetitive recipients detected: The chat history will be cleared by default if a recipient appears more than once. To retain the chat history, please set 'clear_history=false' in the configuration of the repeating agent."
    );
  }
}

/**
 * Create list of Prerequisite (prerequisite_chat_id, chat_id)
 */
function create_async_prerequisites(chatQueue: Array<Record<string, any>>): Array<Prerequisite> {
  const prerequisites: Array<Prerequisite> = [];
  for (const chatInfo of chatQueue) {
    if (!("chat_id" in chatInfo)) {
      throw new Error("Each chat must have a unique id for async multi-chat execution.");
    }
    const chatId = chatInfo.chat_id;
    const preChats = chatInfo.prerequisites || [];
    for (const preChatId of preChats) {
      if (!Number.isInteger(preChatId)) {
        throw new Error("Prerequisite chat id is not an integer.");
      }
      prerequisites.push([chatId, preChatId]);
    }
  }
  return prerequisites;
}

/**
 * Find chat order for async execution based on the prerequisite chats
 *
 * @param chatIds - set of chat ids
 * @param prerequisites - List of Prerequisite (prerequisite_chat_id, chat_id)
 * @returns a list of chat_id in order.
 */
function find_async_chat_order(chatIds: Set<number>, prerequisites: Array<Prerequisite>): Array<number> {
  const edges: Record<number, Set<number>> = {};
  const indegree: Record<number, number> = {};

  for (const pair of prerequisites) {
    const [chat, pre] = [pair[0], pair[1]];
    if (!edges[pre]) {
      edges[pre] = new Set();
    }
    if (!edges[pre].has(chat)) {
      indegree[chat] = (indegree[chat] || 0) + 1;
      edges[pre].add(chat);
    }
  }

  let bfs = Array.from(chatIds).filter((i) => !(i in indegree));
  const chatOrder: Array<number> = [];
  const steps = Object.keys(indegree).length;

  for (let i = 0; i <= steps; i++) {
    if (bfs.length === 0) {
      break;
    }
    chatOrder.push(...bfs);
    const nxt: Array<number> = [];
    for (const node of bfs) {
      if (node in edges) {
        for (const course of edges[node]) {
          indegree[course] -= 1;
          if (indegree[course] === 0) {
            nxt.push(course);
            delete indegree[course];
          }
        }
        delete edges[node];
      }
    }
    bfs = nxt;
  }

  if (Object.keys(indegree).length) {
    return [];
  }
  return chatOrder;
}

/**
 * Process carryover items
 */
function post_process_carryover_item(carryoverItem: any): string {
  if (typeof carryoverItem === "string") {
    return carryoverItem;
  } else if (carryoverItem && typeof carryoverItem === "object" && "content" in carryoverItem) {
    return String(carryoverItem.content);
  } else {
    return String(carryoverItem);
  }
}

/**
 * Post carryover processing
 */
function post_carryover_processing(chatInfo: Record<string, any>): void {
  const iostream = IOStream.getDefault();

  if (!("message" in chatInfo)) {
    console.warn("message is not provided in a chat_queue entry. input() will be called to get the initial message.");
  }

  iostream.send(new post_carryover_processing_message({ chatInfo }));
}

/**
 * Get current system time as string
 */
function system_now_str(): string {
  const ct = new Date();
  return ` System time at ${ct}. `;
}

/**
 * Update ChatResult when async Task for Chat is completed.
 */
function on_chat_future_done(chatFuture: Promise<ChatResult>, chatId: number): void {
  logger.debug(`Update chat ${chatId} result on task completion.` + system_now_str());
  chatFuture.then((chatResult) => {
    chatResult.chatId = chatId;
  });
}

/**
 * Initiate a list of chats.
 *
 * @param chatQueue - A list of dictionaries containing the information about the chats.
 * 
 * Each dictionary should contain the input arguments for
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
               from which to exclude the summaries for carryover. If 'finished_chat_indexes_to_exclude_from_carryover' is not provided or an empty list,
               then summary from all the finished chats will be taken.
 * 
 * @returns a list of ChatResult objects corresponding to the finished chats in the chat_queue.
 */
export async function initiateChats(chatQueue: Array<Record<string, any>>): Promise<Array<ChatResult>> {
  consolidate_chat_info(chatQueue);
  validate_recipients(chatQueue);
  const currentChatQueue = [...chatQueue];
  const finishedChats: Array<ChatResult> = [];

  while (currentChatQueue.length) {
    const chatInfo = currentChatQueue.shift()!;
    let chatCarryover = chatInfo.carryover || [];
    const finishedChatIndexesToExcludeFromCarryover = chatInfo.finished_chat_indexes_to_exclude_from_carryover || [];

    if (typeof chatCarryover === "string") {
      chatCarryover = [chatCarryover];
    }

    chatInfo.carryover = [...chatCarryover, ...finishedChats.filter((_, i) => !finishedChatIndexesToExcludeFromCarryover.includes(i)).map((r) => r.summary)];

    if (!chatInfo.silent) {
      post_carryover_processing(chatInfo);
    }

    const sender = chatInfo.sender;
    const chatRes = await sender.initiateChat(chatInfo);
    finishedChats.push(chatRes);
  }

  return finishedChats;
}

/**
 * Create an async Task for each chat.
 */
async function dependent_chat_future(chatId: number, chatInfo: Record<string, any>, prerequisiteChatFutures: Record<number, Promise<ChatResult>>): Promise<Promise<ChatResult>> {
  logger.debug(`Create Task for chat ${chatId}.` + system_now_str());
  let chatCarryover = chatInfo.carryover || [];
  const finishedChatIndexesToExcludeFromCarryover = chatInfo.finished_chat_indexes_to_exclude_from_carryover || [];
  const finishedChats: Record<number, ChatResult> = {};

  for (const chat in prerequisiteChatFutures) {
    const chatFuture = prerequisiteChatFutures[chat];
    try {
      // wait for prerequisite chat results for the new chat carryover
      finishedChats[chat] = await chatFuture;
    } catch (error) {
      throw new Error(`Chat ${chat} failed: ${error}`);
    }
  }

  if (typeof chatCarryover === "string") {
    chatCarryover = [chatCarryover];
  }

  const data = Object.entries(finishedChats)
    .filter(([chatId]) => !finishedChatIndexesToExcludeFromCarryover.includes(parseInt(chatId)))
    .map(([_, chatResult]) => chatResult.summary);

  chatInfo.carryover = [...chatCarryover, ...data];

  if (!chatInfo.silent) {
    post_carryover_processing(chatInfo);
  }

  const sender = chatInfo.sender;
  const chatResFuture = sender.aInitiateChat(chatInfo);

  chatResFuture.then((result) => {
    on_chat_future_done(Promise.resolve(result), chatId);
  });

  logger.debug(`Task for chat ${chatId} created.` + system_now_str());
  return chatResFuture;
}

/**
 * (async) Initiate a list of chats.
 *
 * @param chatQueue - Please refer to `initiateChats`.
 * @returns a dict of ChatId: ChatResult corresponding to the finished chats in the chat_queue.
 */
export async function a_initiate_chats(chatQueue: Array<Record<string, any>>): Promise<Record<number, ChatResult>> {
  consolidate_chat_info(chatQueue);
  validate_recipients(chatQueue);

  const chatBook: Record<number, Record<string, any>> = {};
  chatQueue.forEach((chatInfo) => {
    chatBook[chatInfo.chat_id] = chatInfo;
  });

  const numChats = new Set(Object.keys(chatBook).map(Number));
  const prerequisites = create_async_prerequisites(chatQueue);
  const chatOrderById = find_async_chat_order(numChats, prerequisites);
  const finishedChatFutures: Record<number, Promise<ChatResult>> = {};

  for (const chatId of chatOrderById) {
    const chatInfo = chatBook[chatId];
    const prerequisiteChatIds = chatInfo.prerequisites || [];
    const preChatFutures: Record<number, Promise<ChatResult>> = {};

    for (const preChatId of prerequisiteChatIds) {
      const preChatFuture = finishedChatFutures[preChatId];
      preChatFutures[preChatId] = preChatFuture;
    }

    const currentChatFuture = await dependent_chat_future(chatId, chatInfo, preChatFutures);
    finishedChatFutures[chatId] = currentChatFuture;
  }

  await Promise.all(Object.values(finishedChatFutures));

  const finishedChats: Record<number, ChatResult> = {};
  for (const chatId in finishedChatFutures) {
    const chatResult = await finishedChatFutures[chatId];
    finishedChats[chatId] = chatResult;
  }

  return finishedChats;
}
