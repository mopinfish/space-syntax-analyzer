# space_syntax_analyzer/exceptions/errors.py
"""
Space Syntax Analyzer カスタム例外クラス
"""


class SpaceSyntaxAnalyzerError(Exception):
    """Space Syntax Analyzer基底例外クラス"""
    pass


class NetworkRetrievalError(SpaceSyntaxAnalyzerError):
    """ネットワーク取得エラー"""
    def __init__(self, message: str, location: str = None, details: dict = None):
        super().__init__(message)
        self.location = location
        self.details = details or {}


class InvalidLocationError(SpaceSyntaxAnalyzerError):
    """無効な位置指定エラー"""
    def __init__(self, message: str, location: str = None):
        super().__init__(message)
        self.location = location


class AnalysisError(SpaceSyntaxAnalyzerError):
    """分析処理エラー"""
    def __init__(self, message: str, analysis_type: str = None, details: dict = None):
        super().__init__(message)
        self.analysis_type = analysis_type
        self.details = details or {}


class VisualizationError(SpaceSyntaxAnalyzerError):
    """可視化処理エラー"""
    def __init__(self, message: str, plot_type: str = None):
        super().__init__(message)
        self.plot_type = plot_type


class DataExportError(SpaceSyntaxAnalyzerError):
    """データエクスポートエラー"""
    def __init__(self, message: str, export_format: str = None, file_path: str = None):
        super().__init__(message)
        self.export_format = export_format
        self.file_path = file_path


class ValidationError(SpaceSyntaxAnalyzerError):
    """データ検証エラー"""
    def __init__(self, message: str, validation_type: str = None, invalid_data = None):
        super().__init__(message)
        self.validation_type = validation_type
        self.invalid_data = invalid_data
