'use client';

import { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, Download, Eye, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import Image from 'next/image';

interface Corner {
  x: number;
  y: number;
}

interface CorrectionResult {
  success: boolean;
  correctedImage?: string;
  corners?: Corner[];
  visualization?: string;
  error?: string;
  isMock?: boolean;  // 添加此字段标记是否为模拟推理
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [result, setResult] = useState<CorrectionResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showVisualization, setShowVisualization] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setResult(null);
    }
  };

  const handleUpload = () => {
    fileInputRef.current?.click();
  };

  const handleCorrect = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('visualization', showVisualization.toString());

    try {
      const response = await fetch('/api/correct', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        setResult({
          success: false,
          error: data.error || 'Processing failed'
        });
      } else {
        setResult(data);
      }
    } catch (error) {
      setResult({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to process image'
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = (imageUrl: string, filename: string) => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = filename;
    link.click();
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold tracking-tight mb-2">
            SmartDoc 文档矫正系统
          </h1>
          <p className="text-muted-foreground">
            基于深度学习的文档四边界定位与透视矫正
          </p>
        </div>

        {/* Main Content */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="w-5 h-5" />
                上传图片
              </CardTitle>
              <CardDescription>
                选择需要矫正的文档图片
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Upload Button */}
              <div className="flex gap-2">
                <Button
                  onClick={handleUpload}
                  className="flex-1"
                  disabled={isProcessing}
                >
                  <ImageIcon className="w-4 h-4 mr-2" />
                  选择图片
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>

              {/* Options */}
              <div className="flex items-center space-x-2">
                <Switch
                  id="visualization"
                  checked={showVisualization}
                  onCheckedChange={setShowVisualization}
                  disabled={isProcessing}
                />
                <Label htmlFor="visualization" className="text-sm">
                  显示检测角点
                </Label>
              </div>

              {/* Preview */}
              {previewUrl && (
                <div className="relative aspect-video bg-slate-100 rounded-lg overflow-hidden">
                  <Image
                    src={previewUrl}
                    alt="Preview"
                    fill
                    className="object-contain"
                  />
                </div>
              )}

              {/* Actions */}
              {selectedFile && (
                <div className="flex gap-2">
                  <Button
                    onClick={handleCorrect}
                    disabled={isProcessing}
                    className="flex-1"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        处理中...
                      </>
                    ) : (
                      '开始矫正'
                    )}
                  </Button>
                  <Button
                    onClick={handleReset}
                    variant="outline"
                    disabled={isProcessing}
                  >
                    重置
                  </Button>
                </div>
              )}

              {/* Error */}
              {result?.error && (
                <Alert variant="destructive">
                  <AlertDescription>
                    {result.error}
                    {result.error.includes('Checkpoint file not found') && (
                      <>
                        <br />
                        <span className="text-xs mt-1 block">
                          提示：请先训练模型或在本地已有模型的情况下使用
                        </span>
                      </>
                    )}
                  </AlertDescription>
                </Alert>
              )}

              {/* Info */}
              {selectedFile && !result && !isProcessing && (
                <Alert>
                  <AlertDescription className="text-sm">
                    注意：首次使用需要训练模型。请查看文档了解如何训练。
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Result Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="w-5 h-5" />
                矫正结果
              </CardTitle>
              <CardDescription>
                矫正后的平展文档
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {result?.success && result.correctedImage ? (
                <>
                  {/* Mock Inference Notice */}
                  {result.isMock && (
                    <Alert>
                      <AlertDescription className="text-sm">
                        ℹ️ 当前使用模拟推理（未检测到训练好的模型）。
                        结果仅为演示，真实模型需要先训练。
                      </AlertDescription>
                    </Alert>
                  )}

                  {/* Corrected Image */}
                  <div className="relative aspect-video bg-slate-100 rounded-lg overflow-hidden">
                    <Image
                      src={result.correctedImage}
                      alt="Corrected"
                      fill
                      className="object-contain"
                    />
                  </div>

                  {/* Visualization */}
                  {showVisualization && result.visualization && (
                    <div className="relative aspect-video bg-slate-100 rounded-lg overflow-hidden">
                      <Image
                        src={result.visualization}
                        alt="Visualization"
                        fill
                        className="object-contain"
                      />
                    </div>
                  )}

                  {/* Corners Info */}
                  {result.corners && (
                    <div className="text-sm text-muted-foreground space-y-1">
                      <div className="font-medium">检测到的角点：</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>左上: ({result.corners[0].x.toFixed(3)}, {result.corners[0].y.toFixed(3)})</div>
                        <div>右上: ({result.corners[1].x.toFixed(3)}, {result.corners[1].y.toFixed(3)})</div>
                        <div>右下: ({result.corners[2].x.toFixed(3)}, {result.corners[2].y.toFixed(3)})</div>
                        <div>左下: ({result.corners[3].x.toFixed(3)}, {result.corners[3].y.toFixed(3)})</div>
                      </div>
                    </div>
                  )}

                  {/* Download */}
                  <Button
                    onClick={() => handleDownload(result.correctedImage!, 'corrected.jpg')}
                    className="w-full"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    下载矫正图片
                  </Button>
                </>
              ) : (
                <div className="aspect-video bg-slate-100 rounded-lg flex items-center justify-center">
                  <p className="text-muted-foreground">
                    等待处理...
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Info Section */}
        <div className="mt-8">
          <Card>
            <CardHeader>
              <CardTitle>系统说明</CardTitle>
            </CardHeader>
            <CardContent className="prose prose-sm max-w-none">
              <ul className="space-y-2">
                <li>
                  <strong>数据集：</strong>SmartDoc 2015 Challenge 1
                </li>
                <li>
                  <strong>模型：</strong>DocAligner - 基于热力图的文档四点检测
                </li>
                <li>
                  <strong>性能指标：</strong>
                  <ul className="ml-4 mt-1 space-y-1">
                    <li>文档区域 IoU ≥ 0.85</li>
                    <li>四点 NME ≤ 0.03</li>
                  </ul>
                </li>
                <li>
                  <strong>功能：</strong>
                  <ul className="ml-4 mt-1 space-y-1">
                    <li>自动检测文档四角点</li>
                    <li>透视变换矫正</li>
                    <li>支持手动微调角点（即将推出）</li>
                    <li>导出JPEG/PDF格式（即将推出）</li>
                  </ul>
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
