/**
 * Full-Text Search via KuzuDB FTS
 *
 * Uses KuzuDB's built-in full-text search indexes for keyword-based search.
 * Always reads from the database (no cached state to drift).
 */
export interface BM25SearchResult {
    filePath: string;
    score: number;
    rank: number;
}
/**
 * Search using KuzuDB's built-in FTS (always fresh, reads from disk)
 *
 * Queries multiple node tables (File, Function, Class, Method) in parallel
 * and merges results by filePath, summing scores for the same file.
 *
 * @param query - Search query string
 * @param limit - Maximum results
 * @param repoId - If provided, queries will be routed via the MCP connection pool
 * @returns Ranked search results from FTS indexes
 */
export declare const searchFTSFromKuzu: (query: string, limit?: number, repoId?: string) => Promise<BM25SearchResult[]>;
