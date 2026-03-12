import { PipelineProgress, PipelineResult } from '../../types/pipeline.js';
export declare const runPipelineFromRepo: (repoPath: string, onProgress: (progress: PipelineProgress) => void) => Promise<PipelineResult>;
