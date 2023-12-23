import json
import urllib.error
import urllib.request
from http.client import HTTPResponse
from io import BytesIO
from pathlib import Path
from typing import Dict, Any
from zipfile import ZipFile, ZIP_DEFLATED
import time

URL = 'https://wdpo.dpieczynski.pl'


def main():
    student_id = '151132'  # Tutaj należy wpisać swój numer indeksu

    data = BytesIO()
    with ZipFile(data, 'w', ZIP_DEFLATED) as zip_file:
        base_path = Path.cwd()
        for file in base_path.rglob('*'):
            if (file.name in ('data.yaml', 'last.py', 'trainmodel.py', 'bestEnum200.pt', 'yolov8n.pt', 'yolov8m.pt', 'yolov8x.pt', 'datasets.zip')
                    or file.is_relative_to(base_path / 'data')
                    or file.is_relative_to(base_path / 'env')
                    or file.is_relative_to(base_path / 'sample_data')
                    or file.is_relative_to(base_path / 'runs')
                    or file.is_relative_to(base_path / '.idea')
                    or file.is_relative_to(base_path / 'datasets1')
                    or file.is_relative_to(base_path / 'datasets')):
                continue

            zip_file.write(file, file.relative_to(base_path))

    data.seek(0)

    try:
        response: HTTPResponse = urllib.request.urlopen(f'{URL}/{student_id}', data.read(), timeout=10000)
        response: Dict[str, Any] = json.loads(response.read())
        print(response)
    except urllib.error.HTTPError as e:
        response: Dict[str, Any] = json.loads(e.read())
        print('ERROR')
        print(response['data'])
        if response['logs']:
            print()
            print('LOGS:')
            print(response['logs'])


if __name__ == '__main__':
    main()
