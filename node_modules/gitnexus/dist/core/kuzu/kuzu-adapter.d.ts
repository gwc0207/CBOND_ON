import kuzu from 'kuzu';
import { KnowledgeGraph } from '../graph/types.js';
export declare const initKuzu: (dbPath: string) => Promise<{
    db: kuzu.Database;
    conn: kuzu.Connection;
}>;
/**
 * Execute multiple queries against one repo DB atomically.
 * While the callback runs, no other request can switch the active DB.
 */
export declare const withKuzuDb: <T>(dbPath: string, operation: () => Promise<T>) => Promise<T>;
export type KuzuProgressCallback = (message: string) => void;
export declare const loadGraphToKuzu: (graph: KnowledgeGraph, repoPath: string, storagePath: string, onProgress?: KuzuProgressCallback) => Promise<{
    success: boolean;
    insertedRels: number;
    skippedRels: number;
    warnings: string[];
}>;
/**
 * Insert a single node to KuzuDB
 * @param label - Node type (File, Function, Class, etc.)
 * @param properties - Node properties
 * @param dbPath - Path to KuzuDB database (optional if already initialized)
 */
export declare const insertNodeToKuzu: (label: string, properties: Record<string, any>, dbPath?: string) => Promise<boolean>;
/**
 * Batch insert multiple nodes to KuzuDB using a single connection
 * @param nodes - Array of {label, properties} to insert
 * @param dbPath - Path to KuzuDB database
 * @returns Object with success count and error count
 */
export declare const batchInsertNodesToKuzu: (nodes: Array<{
    label: string;
    properties: Record<string, any>;
}>, dbPath: string) => Promise<{
    inserted: number;
    failed: number;
}>;
export declare const executeQuery: (cypher: string) => Promise<any[]>;
export declare const executeWithReusedStatement: (cypher: string, paramsList: Array<Record<string, any>>) => Promise<void>;
export declare const getKuzuStats: () => Promise<{
    nodes: number;
    edges: number;
}>;
/**
 * Load cached embeddings from KuzuDB before a rebuild.
 * Returns all embedding vectors so they can be re-inserted after the graph is reloaded,
 * avoiding expensive re-embedding of unchanged nodes.
 */
export declare const loadCachedEmbeddings: () => Promise<{
    embeddingNodeIds: Set<string>;
    embeddings: Array<{
        nodeId: string;
        embedding: number[];
    }>;
}>;
export declare const closeKuzu: () => Promise<void>;
export declare const isKuzuReady: () => boolean;
/**
 * Delete all nodes (and their relationships) for a specific file from KuzuDB
 * @param filePath - The file path to delete nodes for
 * @param dbPath - Optional path to KuzuDB for per-query connection
 * @returns Object with counts of deleted nodes
 */
export declare const deleteNodesForFile: (filePath: string, dbPath?: string) => Promise<{
    deletedNodes: number;
}>;
export declare const getEmbeddingTableName: () => string;
/**
 * Load the FTS extension (required before using FTS functions).
 * Safe to call multiple times — tracks loaded state via module-level ftsLoaded.
 */
export declare const loadFTSExtension: () => Promise<void>;
/**
 * Create a full-text search index on a table
 * @param tableName - The node table name (e.g., 'File', 'CodeSymbol')
 * @param indexName - Name for the FTS index
 * @param properties - List of properties to index (e.g., ['name', 'code'])
 * @param stemmer - Stemming algorithm (default: 'porter')
 */
export declare const createFTSIndex: (tableName: string, indexName: string, properties: string[], stemmer?: string) => Promise<void>;
/**
 * Query a full-text search index
 * @param tableName - The node table name
 * @param indexName - FTS index name
 * @param query - Search query string
 * @param limit - Maximum results
 * @param conjunctive - If true, all terms must match (AND); if false, any term matches (OR)
 * @returns Array of { node properties, score }
 */
export declare const queryFTS: (tableName: string, indexName: string, query: string, limit?: number, conjunctive?: boolean) => Promise<Array<{
    nodeId: string;
    name: string;
    filePath: string;
    score: number;
    [key: string]: any;
}>>;
/**
 * Drop an FTS index
 */
export declare const dropFTSIndex: (tableName: string, indexName: string) => Promise<void>;
