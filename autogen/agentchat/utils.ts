// Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
// SPDX-License-Identifier: MIT

import { Agent } from "./agent";

/**
 * Validates and consolidates chat information
 *
 * @param chat_info Dictionary or list of dictionaries containing chat information
 * @param uniform_sender Optional sender to use for all chat items if provided
 */
export function consolidate_chat_info(chat_info: Record<string, any> | Array<Record<string, any>>, uniform_sender: Agent | null = null): void {
  // Convert a single dictionary to a list containing that dictionary
  if (!Array.isArray(chat_info)) {
    chat_info = [chat_info] as (typeof chat_info)[];
  }

  for (const c of chat_info as Array<Record<string, any>>) {
    let sender: Agent;
    if (uniform_sender === null) {
      if (!("sender" in c)) throw new Error("sender must be provided.");
      sender = c["sender"];
    } else {
      sender = uniform_sender;
    }
    if (!("recipient" in c)) throw new Error("recipient must be provided.");
    const summary_method = c["summary_method"];
    if (summary_method !== undefined && summary_method !== null && typeof summary_method !== "function" && summary_method !== "last_msg" && summary_method !== "reflection_with_llm") {
      throw new Error("summary_method must be a string chosen from 'reflection_with_llm' or 'last_msg' or a callable, or None.");
    }
    if (summary_method === "reflection_with_llm") {
      if (!(sender.client !== undefined && sender.client !== null) && !(c["recipient"].client !== undefined && c["recipient"].client !== null)) {
        throw new Error("llm client must be set in either the recipient or sender when summary_method is reflection_with_llm.");
      }
    }
  }
}

/**
 * Gathers usage summary from all agents
 *
 * @param agents List of agents to gather usage from
 * @returns Dictionary containing usage summaries
 */
export function gather_usage_summary(agents: Array<Agent>): Record<string, Record<string, any>> {
  /**
   * Aggregates usage summary information
   *
   * @param usage_summary Summary to update
   * @param agent_summary Summary to add
   */
  function aggregate_summary(usage_summary: Record<string, any>, agent_summary: Record<string, any> | null): void {
    if (agent_summary === null) {
      return;
    }

    usage_summary["total_cost"] += agent_summary["total_cost"] || 0;

    for (const [model, data] of Object.entries(agent_summary)) {
      if (model !== "total_cost") {
        if (!(model in usage_summary)) {
          usage_summary[model] = { ...data };
        } else {
          usage_summary[model]["cost"] += data["cost"] || 0;
          usage_summary[model]["prompt_tokens"] += data["prompt_tokens"] || 0;
          usage_summary[model]["completion_tokens"] += data["completion_tokens"] || 0;
          usage_summary[model]["total_tokens"] += data["total_tokens"] || 0;
        }
      }
    }
  }

  const usage_including_cached_inference = { total_cost: 0 };
  const usage_excluding_cached_inference = { total_cost: 0 };

  for (const agent of agents) {
    if (agent.client) {
      aggregate_summary(usage_including_cached_inference, agent.client.total_usage_summary);
      aggregate_summary(usage_excluding_cached_inference, agent.client.actual_usage_summary);
    }
  }

  return {
    usage_including_cached_inference: usage_including_cached_inference,
    usage_excluding_cached_inference: usage_excluding_cached_inference,
  };
}

/**
 * Parses HTML style tags from message contents
 *
 * @param tag The HTML style tag to be parsed
 * @param content The message content to parse
 * @returns List of parsed tag objects
 */
export function parse_tags_from_content(tag: string, content: string | Array<Record<string, any>>): Array<Record<string, any>> {
  const results: Array<Record<string, any>> = [];

  if (typeof content === "string") {
    results.push(..._parse_tags_from_text(tag, content));
  }
  // Handles case for multimodal messages
  else if (Array.isArray(content)) {
    for (const item of content) {
      if (item["type"] === "text") {
        results.push(..._parse_tags_from_text(tag, item["text"]));
      }
    }
  } else {
    throw new Error(`content must be string or array, but got ${typeof content}`);
  }

  return results;
}

/**
 * Parses tags from a text string
 *
 * @param tag The tag name to look for
 * @param text The text to parse
 * @returns List of parsed tag objects
 */
function _parse_tags_from_text(tag: string, text: string): Array<Record<string, any>> {
  const pattern = new RegExp(`<${tag} (.*?)>`, "g");
  const results: Array<Record<string, any>> = [];

  let match;
  while ((match = pattern.exec(text)) !== null) {
    const tag_attr = match[1].trim();
    const attr = _parse_attributes_from_tags(tag_attr);

    results.push({ tag: tag, attr: attr, match: match });
  }

  return results;
}

/**
 * Parses attributes from tag content
 *
 * @param tag_content The tag content to parse
 * @returns Dictionary of parsed attributes
 */
function _parse_attributes_from_tags(tag_content: string): Record<string, string> {
  const pattern = /([^ ]+)/g;
  const attrs = [];

  let match;
  while ((match = pattern.exec(tag_content)) !== null) {
    attrs.push(match[1]);
  }

  const reconstructed_attrs = _reconstruct_attributes(attrs);
  const content: Record<string, string> = {};

  /**
   * Adds or appends a value to the src attribute
   *
   * @param content The content object to modify
   * @param value The value to add
   */
  function _append_src_value(content: Record<string, string>, value: any): void {
    if ("src" in content) {
      content["src"] += ` ${value}`;
    } else {
      content["src"] = value;
    }
  }

  for (const attr of reconstructed_attrs) {
    if (!attr.includes("=")) {
      _append_src_value(content, attr);
      continue;
    }

    const [key, value] = attr.split("=", 2);
    if (value.startsWith("'") || value.startsWith('"')) {
      content[key] = value.substring(1, value.length - 1); // remove quotes
    } else {
      _append_src_value(content, attr);
    }
  }

  return content;
}

/**
 * Reconstructs attributes from a list of strings
 *
 * @param attrs List of attribute strings
 * @returns Reconstructed list of attribute strings
 */
function _reconstruct_attributes(attrs: Array<string>): Array<string> {
  /**
   * Checks if a string is a valid attribute
   *
   * @param attr The attribute string to check
   * @returns Whether the string is a valid attribute
   */
  function is_attr(attr: string): boolean {
    if (attr.includes("=")) {
      const [, value] = attr.split("=", 2);
      if (value.startsWith("'") || value.startsWith('"')) {
        return true;
      }
    }
    return false;
  }

  const reconstructed: Array<string> = [];
  let found_attr = false;

  for (const attr of attrs) {
    if (is_attr(attr)) {
      reconstructed.push(attr);
      found_attr = true;
    } else {
      if (found_attr) {
        reconstructed[reconstructed.length - 1] += ` ${attr}`;
      } else if (reconstructed.length > 0) {
        reconstructed[reconstructed.length - 1] += ` ${attr}`;
      } else {
        reconstructed.push(attr);
      }
    }
  }

  return reconstructed;
}

/**
 * A class to evaluate logical expressions using context variables
 */
export class ContextExpression {
  private expression: string;
  private _variable_names: Array<string>;
  private _python_expr: string;

  /**
   * Creates a new ContextExpression
   *
   * @param expression The logical expression to evaluate
   */
  constructor(expression: string) {
    this.expression = expression;

    try {
      // Extract variable references
      this._variable_names = this._extract_variable_names(this.expression);

      // Convert symbolic operators to JavaScript syntax
      this._python_expr = this._convert_to_js_syntax(this.expression);

      // Validate the expression
      this._validate_expression(this._python_expr);
    } catch (e) {
      if (e instanceof SyntaxError) {
        throw new SyntaxError(`Invalid expression syntax in '${this.expression}': ${e.message}`);
      } else {
        throw new Error(`Error validating expression '${this.expression}': ${String(e)}`);
      }
    }
  }

  /**
   * Extracts variable names from the expression
   *
   * @param expr The expression to extract from
   * @returns List of variable names
   */
  private _extract_variable_names(expr: string): Array<string> {
    const matches = expr.match(/\${([^}]*)}/g) || [];
    return matches.map((match) => match.slice(2, -1));
  }

  /**
   * Converts Python-like syntax to JavaScript syntax
   *
   * @param expr The expression to convert
   * @returns Converted expression
   */
  private _convert_to_js_syntax(expr: string): string {
    // Store string literals temporarily
    const string_literals: Array<string> = [];

    const expr_without_strings = expr.replace(/'[^']*'|"[^"]*"/g, (match) => {
      string_literals.push(match);
      return `__STRING_LITERAL_${string_literals.length - 1}__`;
    });

    // Handle NOT operator (!)
    let converted = expr_without_strings.replace(/!\s*(\${|\()/g, "!$1");

    // Handle AND and OR operators
    converted = converted.replace(/\s+&\s+/g, " && ");
    converted = converted.replace(/\s+\|\s+/g, " || ");

    // Restore string literals
    for (let i = 0; i < string_literals.length; i++) {
      converted = converted.replace(`__STRING_LITERAL_${i}__`, string_literals[i]);
    }

    return converted;
  }

  /**
   * Validates the expression for allowed operations
   *
   * @param expr The expression to validate
   */
  private _validate_expression(expr: string): void {
    // In TypeScript, we can't use the Python AST for validation
    // We'll do a simpler validation to check for obvious issues

    // Check for balanced parentheses
    let openParens = 0;
    for (const char of expr) {
      if (char === "(") openParens++;
      if (char === ")") openParens--;
      if (openParens < 0) throw new SyntaxError("Unbalanced parentheses");
    }
    if (openParens !== 0) throw new SyntaxError("Unbalanced parentheses");

    // Check for disallowed JavaScript features
    const disallowed = ["function", "=>", "class", "new", "import", "export", "delete", "void", "instanceof", "typeof", "try", "catch"];

    for (const keyword of disallowed) {
      if (expr.includes(keyword)) {
        throw new Error(`Disallowed keyword in expression: ${keyword}`);
      }
    }
  }

  /**
   * Evaluates the expression with the given context variables
   *
   * @param context_variables Dictionary of variables to use in evaluation
   * @returns Result of evaluation
   */
  evaluate(context_variables: Record<string, any>): boolean {
    let eval_expr = this._python_expr;

    // Process len() functions
    const len_pattern = /len\(\${([^}]*)}\)/g;
    let len_match;

    while ((len_match = len_pattern.exec(eval_expr)) !== null) {
      const var_name = len_match[1];
      const var_value = context_variables[var_name] || [];

      // Calculate length
      let length_value = 0;
      try {
        length_value = var_value.length || 0;
      } catch (e) {
        // If length isn't available, use 0
        length_value = 0;
      }

      // Replace the len() expression with the length value
      eval_expr = eval_expr.replace(len_match[0], String(length_value));
    }

    // Replace variable references with their values
    for (const var_name of this._variable_names) {
      // Skip variables already processed in len() expressions
      if (eval_expr.includes(`\${${var_name}}`)) {
        // Get the value, defaulting to false if not found
        const var_value = context_variables[var_name] !== undefined ? context_variables[var_name] : false;

        // Format the value based on its type
        let formatted_value;

        if (typeof var_value === "boolean" || typeof var_value === "number") {
          formatted_value = String(var_value);
        } else if (typeof var_value === "string") {
          formatted_value = `'${var_value}'`; // Quote strings
        } else if (Array.isArray(var_value) || typeof var_value === "object") {
          // For collections, convert to their boolean evaluation
          formatted_value = String(Boolean(var_value));
        } else {
          formatted_value = String(var_value);
        }

        // Replace the variable reference with the formatted value
        eval_expr = eval_expr.replace(`\${${var_name}}`, formatted_value);
      }
    }

    try {
      // Use Function constructor for safer evaluation than eval
      return new Function(`return ${eval_expr}`)() as boolean;
    } catch (e) {
      throw new Error(`Error evaluating expression '${this.expression}': ${String(e)}`);
    }
  }

  toString(): string {
    return `ContextExpression('${this.expression}')`;
  }
}
