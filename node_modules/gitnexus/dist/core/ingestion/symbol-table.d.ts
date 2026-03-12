export interface SymbolDefinition {
    nodeId: string;
    filePath: string;
    type: string;
}
export interface SymbolTable {
    /**
     * Register a new symbol definition
     */
    add: (filePath: string, name: string, nodeId: string, type: string) => void;
    /**
     * High Confidence: Look for a symbol specifically inside a file
     * Returns the Node ID if found
     */
    lookupExact: (filePath: string, name: string) => string | undefined;
    /**
     * Low Confidence: Look for a symbol anywhere in the project
     * Used when imports are missing or for framework magic
     */
    lookupFuzzy: (name: string) => SymbolDefinition[];
    /**
     * Debugging: See how many symbols are tracked
     */
    getStats: () => {
        fileCount: number;
        globalSymbolCount: number;
    };
    /**
     * Cleanup memory
     */
    clear: () => void;
}
export declare const createSymbolTable: () => SymbolTable;
