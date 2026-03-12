import { KnowledgeGraph } from '../graph/types.js';
import { ASTCache } from './ast-cache.js';
import type { ExtractedImport } from './workers/parse-worker.js';
export type ImportMap = Map<string, Set<string>>;
export declare const createImportMap: () => ImportMap;
/** Pre-built lookup structures for import resolution. Build once, reuse across chunks. */
export interface ImportResolutionContext {
    allFilePaths: Set<string>;
    allFileList: string[];
    normalizedFileList: string[];
    suffixIndex: SuffixIndex;
    resolveCache: Map<string, string | null>;
}
export declare function buildImportResolutionContext(allPaths: string[]): ImportResolutionContext;
/**
 * Build a suffix index for O(1) endsWith lookups.
 * Maps every possible path suffix to its original file path.
 * e.g. for "src/com/example/Foo.java":
 *   "Foo.java" -> "src/com/example/Foo.java"
 *   "example/Foo.java" -> "src/com/example/Foo.java"
 *   "com/example/Foo.java" -> "src/com/example/Foo.java"
 *   etc.
 */
export interface SuffixIndex {
    /** Exact suffix lookup (case-sensitive) */
    get(suffix: string): string | undefined;
    /** Case-insensitive suffix lookup */
    getInsensitive(suffix: string): string | undefined;
    /** Get all files in a directory suffix */
    getFilesInDir(dirSuffix: string, extension: string): string[];
}
export declare const processImports: (graph: KnowledgeGraph, files: {
    path: string;
    content: string;
}[], astCache: ASTCache, importMap: ImportMap, onProgress?: (current: number, total: number) => void, repoRoot?: string, allPaths?: string[]) => Promise<void>;
export declare const processImportsFromExtracted: (graph: KnowledgeGraph, files: {
    path: string;
}[], extractedImports: ExtractedImport[], importMap: ImportMap, onProgress?: (current: number, total: number) => void, repoRoot?: string, prebuiltCtx?: ImportResolutionContext) => Promise<void>;
