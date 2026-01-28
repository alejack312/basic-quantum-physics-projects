# Installing FFmpeg for Matplotlib Animations

Matplotlib is a powerful plotting library for Python that allows you to create static, animated, and interactive visualizations. 
When creating animations, Matplotlib can utilize FFmpeg, a popular multimedia framework, to save animations in various video formats. 
This guide will walk you through the steps to install FFmpeg and configure Matplotlib to use it for saving animations as video files.

## 🛠️ Installation of FFmpeg per OS

### Windows

On Windows, one must manually download a compiled FFmpeg build and make sure its bin/ folder is in the system’s PATH environment variable. Example procedure:

1. Go to the [FFmpeg official website](https://ffmpeg.org/download.html).
2. Click on the "Windows" logo to navigate to the Windows download section.
3. Extract the ZIP/7z archive to a folder, e.g. C:\ffmpeg\.
4. Add the bin/ folder to your system’s PATH environment variable:
   - Open the Start Menu and search for "Environment Variables".
   - Click on "Edit the system environment variables".
   - In the System Properties window, click on the "Environment Variables" button.
   - In the Environment Variables window, find the "Path" variable in the "System variables" section and click "Edit".
   - Click "New" and add the path to the bin/ folder (e.g., C:\ffmpeg\bin\).
   - Click "OK" to close all windows.
5. Open a new Command Prompt and type `ffmpeg -version` to verify the installation. You should see FFmpeg version information if the installation was successful.
6. Now, Matplotlib should be able to find FFmpeg automatically. You can test it by importing Matplotlib and checking the animation writers:
```python
import matplotlib.animation as animation
import shutil
 
print(shutil.which("ffmpeg"))  # Should print the path to ffmpeg
print(animation.writers.list())  # Should include 'ffmpeg'
```

If `shutil.which("ffmpeg")` returns `None`, you may need to manually set the path in your script (e.g., `animation.FFMpegWriter.ffmpeg_path = 'C:\\ffmpeg\\bin\\ffmpeg.exe'`).

### macOS

On macOS, the easiest way to install FFmpeg is through Homebrew, a popular package manager for macOS. If you don't have Homebrew installed, you can install it by following the instructions on the [Homebrew website](https://brew.sh/).
1. Open the Terminal application.
2. Install FFmpeg using Homebrew by running the following command:
   ```bash
   brew install ffmpeg
   ```
3. After the installation is complete, verify that FFmpeg is installed by running:
   ```bash
    ffmpeg -version
    ```
4. Matplotlib should automatically detect FFmpeg. You can test it by running the same Python code as shown in the Windows section.

If you don't want to use Homebrew, you can also download precompiled binaries from the [FFmpeg website](https://ffmpeg.org/download.html#build-mac).
Adding the binary to your PATH can be done by editing your shell profile:
1. Open your terminal and edit your shell profile (e.g., `~/.bash_profile`, `~/.zshrc`, etc.):
   ```bash
   nano ~/.zshrc  # or ~/.bash_profile
   ```
2. Add the following line to include the FFmpeg binary path:
   ```bash
    export PATH="/path/to/ffmpeg/bin:$PATH"
    ```
3. Save the file and reload your shell profile:
    ```bash
    source ~/.zshrc  # or source ~/.bash_profile
    ```
4. Verify the installation by running `ffmpeg -version` in the terminal.

### Linux
On Linux, FFmpeg can typically be installed using the package manager that comes with your distribution. Here are the commands for some common distributions:
- **Ubuntu/Debian**:
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```
- **Fedora**:
   ```bash
    sudo dnf install ffmpeg
    ```
- **Arch Linux**:
    ```bash
     sudo pacman -S ffmpeg
     ```
After installation, verify that FFmpeg is installed by running:
```bash
ffmpeg -version
```