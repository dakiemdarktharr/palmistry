"""Display the local Palmistry UI in a native Windows application window."""
import hashlib
from pathlib import Path
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from PySide6.QtWebEngineCore import QWebEngineLoadingInfo, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView


APP_URL = "http://127.0.0.1:8501/keypoints"
INSTANCE_NAME = "PalmistryDesktop-" + hashlib.sha256(
    str(Path(__file__).resolve().parent).lower().encode("utf-8")
).hexdigest()[:16]


class AppView(QWebEngineView):
    def createWindow(self, window_type):
        # Review pages and full-size images also stay inside the desktop app.
        window = AppWindow(self.window())
        window.show()
        return window.view


class AppWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Palmistry Live Camera")
        self.resize(1440, 940)
        self.setMinimumSize(800, 600)
        self.view = AppView(self)
        self.setCentralWidget(self.view)
        self.view.page().loadingChanged.connect(self._loaded)
        self._back = QShortcut(QKeySequence("Alt+Left"), self)
        self._back.activated.connect(self.view.back)
        self._reload = QShortcut(QKeySequence("Ctrl+R"), self)
        self._reload.activated.connect(self.view.reload)

    def _loaded(self, info):
        if info.isDownload():
            return
        if info.status() == QWebEngineLoadingInfo.LoadStatus.LoadFailedStatus:
            self.statusBar().showMessage(
                "Không tải được giao diện Palmistry. Nhấn Ctrl+R để thử lại."
            )
        elif info.status() == QWebEngineLoadingInfo.LoadStatus.LoadSucceededStatus:
            self.statusBar().hide()

    def bring_to_front(self):
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()


def save_download(window, download):
    folder = Path(download.downloadDirectory())
    destination, _ = QFileDialog.getSaveFileName(
        window, "Lưu file Palmistry", str(folder / download.downloadFileName())
    )
    if not destination:
        download.cancel()
        return
    path = Path(destination)
    download.setDownloadDirectory(str(path.parent))
    download.setDownloadFileName(path.name)
    download.accept()


def main():
    application = QApplication(sys.argv)
    application.setApplicationName("Palmistry Live Camera")
    application.setOrganizationName("Palmistry")

    connection = QLocalSocket()
    connection.connectToServer(INSTANCE_NAME)
    if connection.waitForConnected(500):
        connection.write(b"show")
        connection.flush()
        connection.waitForBytesWritten(500)
        connection.disconnectFromServer()
        return 0

    server = QLocalServer()
    if not server.listen(INSTANCE_NAME):
        # Another launch may have won the race while we were connecting.
        connection.abort()
        connection.connectToServer(INSTANCE_NAME)
        if connection.waitForConnected(1000):
            connection.write(b"show")
            connection.flush()
            connection.waitForBytesWritten(500)
            return 0
        QMessageBox.critical(None, "Palmistry", server.errorString())
        return 1

    window = AppWindow()

    def activate_existing_window():
        while server.hasPendingConnections():
            client = server.nextPendingConnection()
            client.disconnected.connect(client.deleteLater)
            window.bring_to_front()
            client.disconnectFromServer()

    server.newConnection.connect(activate_existing_window)
    QWebEngineProfile.defaultProfile().downloadRequested.connect(
        lambda download: save_download(window, download)
    )
    window.view.setUrl(QUrl(APP_URL))
    window.show()
    try:
        return application.exec()
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(main())
