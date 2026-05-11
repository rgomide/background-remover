import os from 'node:os';

export function trySetProcessPriority(priority) {
  if (!Number.isInteger(priority)) {
    return;
  }
  try {
    os.setPriority(priority);
    console.log(`Process priority set to ${priority}`);
  } catch (error) {
    console.warn(`Could not set process priority to ${priority}: ${error.message}`);
  }
}
