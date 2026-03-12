/**
 * Heritage Processor
 *
 * Extracts class inheritance relationships:
 * - EXTENDS: Class extends another Class (TS, JS, Python)
 * - IMPLEMENTS: Class implements an Interface (TS only)
 */
import { KnowledgeGraph } from '../graph/types.js';
import { ASTCache } from './ast-cache.js';
import { SymbolTable } from './symbol-table.js';
import type { ExtractedHeritage } from './workers/parse-worker.js';
export declare const processHeritage: (graph: KnowledgeGraph, files: {
    path: string;
    content: string;
}[], astCache: ASTCache, symbolTable: SymbolTable, onProgress?: (current: number, total: number) => void) => Promise<void>;
/**
 * Fast path: resolve pre-extracted heritage from workers.
 * No AST parsing — workers already extracted className + parentName + kind.
 */
export declare const processHeritageFromExtracted: (graph: KnowledgeGraph, extractedHeritage: ExtractedHeritage[], symbolTable: SymbolTable, onProgress?: (current: number, total: number) => void) => Promise<void>;
