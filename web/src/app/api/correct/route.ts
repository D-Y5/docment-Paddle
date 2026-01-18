import { NextRequest, NextResponse } from 'next/server';
import { writeFile, unlink } from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';

export const runtime = 'nodejs';
export const maxDuration = 60;

interface Corner {
  x: number;
  y: number;
}

interface CorrectResponse {
  success: boolean;
  correctedImage?: string;
  corners?: Corner[];
  visualization?: string;
  error?: string;
}

/**
 * POST /api/correct
 * 处理文档矫正请求
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const saveVisualization = formData.get('visualization') === 'true';

    if (!file) {
      return NextResponse.json<CorrectResponse>({
        success: false,
        error: 'No file uploaded'
      }, { status: 400 });
    }

    // 保存上传的文件
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const uploadDir = path.join(process.cwd(), 'tmp', 'uploads');
    const outputDir = path.join(process.cwd(), 'tmp', 'outputs');

    await ensureDir(uploadDir);
    await ensureDir(outputDir);

    const timestamp = Date.now();
    const inputFileName = `input_${timestamp}.jpg`;
    const outputFileName = `output_${timestamp}.jpg`;
    const visFileName = `vis_${timestamp}.jpg`;

    const inputPath = path.join(uploadDir, inputFileName);
    const outputPath = path.join(outputDir, outputFileName);
    const visPath = path.join(outputDir, visFileName);

    await writeFile(inputPath, buffer);

    // 调用Python推理脚本
    const inferenceResult = await runInference(inputPath, outputPath, visPath, saveVisualization);

    if (!inferenceResult.success) {
      return NextResponse.json<CorrectResponse>({
        success: false,
        error: inferenceResult.error
      }, { status: 500 });
    }

    // 清理临时文件
    await unlink(inputPath).catch(() => {});

    // 返回结果
    const response: CorrectResponse = {
      success: true,
      correctedImage: inferenceResult.correctedImage,
      corners: inferenceResult.corners
    };

    if (saveVisualization && inferenceResult.visualization) {
      response.visualization = inferenceResult.visualization;
    }

    return NextResponse.json(response);

  } catch (error) {
    console.error('Error in /api/correct:', error);
    return NextResponse.json<CorrectResponse>({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

/**
 * 运行Python推理脚本
 */
async function runInference(
  inputPath: string,
  outputPath: string,
  visPath: string,
  saveVisualization: boolean
): Promise<{
  success: boolean;
  correctedImage?: string;
  corners?: Corner[];
  visualization?: string;
  error?: string;
}> {
  return new Promise((resolve) => {
    const projectRoot = path.join(process.cwd(), '..');
    const pythonScript = path.join(projectRoot, 'scripts', 'inference.py');
    const mockScript = path.join(projectRoot, 'scripts', 'test_mock_inference.py');
    const checkpoint = path.join(projectRoot, 'outputs', 'docaligner_20240115', 'checkpoint_best.pth');

    // 检查是否有训练好的模型
    const fs = require('fs');
    const useMock = !fs.existsSync(checkpoint);

    // 根据是否有模型选择脚本
    const scriptToRun = useMock ? mockScript : pythonScript;

    const args = [
      scriptToRun,
      '--input', inputPath,
      '--output', outputPath,
      '--save-corners'
    ];

    if (saveVisualization) {
      args.push('--save-visualization');
    }

    const python = spawn('python', args, { cwd: projectRoot });

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('close', async (code) => {
      if (code !== 0) {
        console.error('Inference failed:', stderr);
        resolve({
          success: false,
          error: stderr || 'Inference failed'
        });
        return;
      }

      // 读取角点标注
      const cornersPath = outputPath.replace('.jpg', '_corners.json');
      let corners: Corner[] = [];

      try {
        const fs = await import('fs/promises');
        const cornersData = await fs.readFile(cornersPath, 'utf-8');
        const cornersJson = JSON.parse(cornersData);

        // 添加安全检查
        if (cornersJson && cornersJson.corners && Array.isArray(cornersJson.corners)) {
          corners = cornersJson.corners.map((c: number[]) => ({
            x: c[0],
            y: c[1]
          }));
        }

        await fs.unlink(cornersPath).catch(() => {});
      } catch (error) {
        console.error('Error reading corners:', error);
      }

      // 读取图像并转换为base64
      const fs = await import('fs/promises');
      const correctedImage = await fs.readFile(outputPath);
      const correctedImageBase64 = correctedImage.toString('base64');

      let visualizationBase64 = '';
      if (saveVisualization) {
        try {
          const visImage = await fs.readFile(visPath);
          visualizationBase64 = visImage.toString('base64');
          await fs.unlink(visPath).catch(() => {});
        } catch (error) {
          console.error('Error reading visualization:', error);
        }
      }

      // 清理输出文件
      await fs.unlink(outputPath).catch(() => {});

      resolve({
        success: true,
        correctedImage: `data:image/jpeg;base64,${correctedImageBase64}`,
        corners,
        visualization: saveVisualization ? `data:image/jpeg;base64,${visualizationBase64}` : undefined
      });
    });

    python.on('error', (error) => {
      console.error('Failed to start Python process:', error);
      resolve({
        success: false,
        error: error.message
      });
    });
  });
}

/**
 * 确保目录存在
 */
async function ensureDir(dir: string): Promise<void> {
  const fs = await import('fs/promises');
  await fs.mkdir(dir, { recursive: true });
}
