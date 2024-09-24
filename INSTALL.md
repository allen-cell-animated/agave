# Installation Instructions

## Windows

Download the installer `agave-1.7.0-win64.exe` and run it.

It will take you through the installation process. In most cases you can accept all the default settings.

Once installed, you can run the application from the start menu.
The first time, you will get a warning from Windows Defender SmartScreen. Click on "More info" and then "Run anyway".

## MacOS

We provide separate installers for MacOS with Apple Silicon (Macs with M1-M4 processors) and MacOS with Intel processors.

The Intel (x86-64) version should work on all Macs. However, you will get the best performance by choosing the matching installer. You can check your Mac's processor architecture by clicking the Apple menu, selecting About This Mac. Look for the Processor Name.

### MacOS on Apple Silicon (Macs with M1-M4 processors)

Download the installer `agave-1.7.0-macos-arm64.dmg` and open it.

Drag the agave icon to the Applications folder.

You can now run the application from the Applications folder.
The first time, you will get a warning that the application is from an unidentified developer. Right-click on the application and select "Open". You will get a warning that the application is from an unidentified developer. Click on "Open".
If you get a warning that the application is damaged, you can run the following command in the terminal to remove the quarantine attribute:

```
xattr -d com.apple.quarantine /Applications/agave.app
```

After this, you should be able to run the application.

### MacOS on Intel processors

Download the installer `agave-1.7.0-macos-x86-64.dmg` and open it.

Drag the agave icon to the Applications folder.

You can now run the application from the Applications folder.
The first time, you will get a warning that the application is from an unidentified developer. Right-click on the application and select "Open". You will get a warning that the application is from an unidentified developer. Click on "Open".
