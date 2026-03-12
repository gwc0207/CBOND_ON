export const createSymbolTable = () => {
    // 1. File-Specific Index (The "Good" one)
    // Structure: FilePath -> (SymbolName -> NodeID)
    const fileIndex = new Map();
    // 2. Global Reverse Index (The "Backup")
    // Structure: SymbolName -> [List of Definitions]
    const globalIndex = new Map();
    const add = (filePath, name, nodeId, type) => {
        // A. Add to File Index
        if (!fileIndex.has(filePath)) {
            fileIndex.set(filePath, new Map());
        }
        fileIndex.get(filePath).set(name, nodeId);
        // B. Add to Global Index
        if (!globalIndex.has(name)) {
            globalIndex.set(name, []);
        }
        globalIndex.get(name).push({ nodeId, filePath, type });
    };
    const lookupExact = (filePath, name) => {
        const fileSymbols = fileIndex.get(filePath);
        if (!fileSymbols)
            return undefined;
        return fileSymbols.get(name);
    };
    const lookupFuzzy = (name) => {
        return globalIndex.get(name) || [];
    };
    const getStats = () => ({
        fileCount: fileIndex.size,
        globalSymbolCount: globalIndex.size
    });
    const clear = () => {
        fileIndex.clear();
        globalIndex.clear();
    };
    return { add, lookupExact, lookupFuzzy, getStats, clear };
};
