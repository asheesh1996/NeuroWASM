// camera.ts

export class CameraManager {
  private videoElement: HTMLVideoElement;
  private currentStream: MediaStream | null = null;
  private devices: MediaDeviceInfo[] = [];

  constructor(videoElementId: string) {
    this.videoElement = document.getElementById(videoElementId) as HTMLVideoElement;
    if (!this.videoElement) {
      throw new Error(`Video element with ID '${videoElementId}' not found.`);
    }
  }

  // Load available camera devices
  async getAvailableCameras(): Promise<MediaDeviceInfo[]> {
    try {
      // Prompt for permissions first to ensure device labels are visible
      const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
      tempStream.getTracks().forEach(track => track.stop());

      const allDevices = await navigator.mediaDevices.enumerateDevices();
      this.devices = allDevices.filter(device => device.kind === 'videoinput');
      return this.devices;
    } catch (error) {
      console.error('Error fetching cameras', error);
      return [];
    }
  }

  // Start specific camera by deviceId, or default camera if no ID provided
  async startCamera(deviceId?: string): Promise<boolean> {
    this.stopCamera();

    try {
      const constraints: MediaStreamConstraints = {
        video: deviceId 
          ? { deviceId: { exact: deviceId } } 
          : { facingMode: 'environment' } // Try to get back camera initially
      };

      this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      this.videoElement.srcObject = this.currentStream;
      
      // Return true when the video starts playing
      return new Promise<boolean>((resolve) => {
        this.videoElement.onloadedmetadata = () => {
          this.videoElement.play();
          resolve(true);
        };
      });

    } catch (error) {
      console.error('Error starting camera', error);
      return false;
    }
  }

  // Stop currently running stream
  stopCamera() {
    if (this.currentStream) {
      this.currentStream.getTracks().forEach(track => track.stop());
      this.currentStream = null;
    }
  }

  // Get raw video element for canvas drawing
  getVideoElement(): HTMLVideoElement {
    return this.videoElement;
  }
}
