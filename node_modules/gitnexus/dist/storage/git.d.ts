export declare const isGitRepo: (repoPath: string) => boolean;
export declare const getCurrentCommit: (repoPath: string) => string;
/**
 * Find the git repository root from any path inside the repo
 */
export declare const getGitRoot: (fromPath: string) => string | null;
