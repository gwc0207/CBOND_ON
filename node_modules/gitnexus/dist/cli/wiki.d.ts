/**
 * Wiki Command
 *
 * Generates repository documentation from the knowledge graph.
 * Usage: gitnexus wiki [path] [options]
 */
export interface WikiCommandOptions {
    force?: boolean;
    model?: string;
    baseUrl?: string;
    apiKey?: string;
    concurrency?: string;
    gist?: boolean;
}
export declare const wikiCommand: (inputPath?: string, options?: WikiCommandOptions) => Promise<void>;
