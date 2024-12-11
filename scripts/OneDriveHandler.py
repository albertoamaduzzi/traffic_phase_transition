import onedrivesdk
from onedrivesdk.helpers import GetAuthCodeServer
import os

class OneDriveHandler:
    """
        This class handles connection to OneDrive
    """
    def __init__(self,config):
        assert 'client_id' in config.keys(), "client_id not in config"
        assert 'scopes' in config.keys(), "scopes not in config"
        assert 'base_dir_project_one_drive' in config.keys(), "base_dir_project_one_drive not in config"
        assert 'auth_url' in config.keys(), "auth_url not in config"
        assert 'redirect_uri' in config.keys(), "redirect_uri not in config"
        assert 'client_secret' in config, "client_secret not in config"
        self.Client = onedrivesdk.get_default_client(client_id=config['client_id'],
                                                     scopes=config['scopes'])
        self.code = GetAuthCodeServer.get_auth_code(config['auth_url'], config['redirect_uri'])
        self.Client.auth_provider.authenticate(self.code, config['redirect_uri'], config['client_secret'])
        self.BaseDirOneDrive = config["base_dir_project_one_drive"]


    def UploadFile2OneDrive(self, local_file_path, onedrive_folder):
        """
        Upload a file to OneDrive.
        
        :param client: Authenticated OneDrive client.
        :param local_file_path: Path to the local file to be uploaded.
        :param onedrive_folder: OneDrive folder where the file will be uploaded.
        """
        file_name = os.path.basename(local_file_path)
        self.Client.item(drive='me', path=onedrive_folder).children[file_name].upload(local_file_path)
        print(f"File {local_file_path} has been uploaded to OneDrive folder {onedrive_folder}.")

    def DownloadFileFromOneDrive(self, onedrive_file_path, local_folder):
        """
        Download a file from OneDrive.
        
        :param client: Authenticated OneDrive client.
        :param onedrive_file_path: Path to the OneDrive file to be downloaded.
        :param local_folder: Local folder where the file will be downloaded.
        """
        file_name = os.path.basename(onedrive_file_path)
        local_file_path = os.path.join(local_folder, file_name)
        self.Client.item(drive='me', path=onedrive_file_path).download(local_file_path)
        print(f"File {onedrive_file_path} has been downloaded to {local_file_path}.")
        return local_file_path