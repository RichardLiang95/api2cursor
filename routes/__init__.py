"""路由注册"""


def register_routes(app):
    """将所有路由蓝图注册到 Flask 应用"""
    from routes.chat import bp as chat_bp
    from routes.responses import bp as responses_bp
    from routes.messages import bp as messages_bp
    from routes.admin import bp as admin_bp

    app.register_blueprint(chat_bp)
    app.register_blueprint(responses_bp)
    app.register_blueprint(messages_bp)
    app.register_blueprint(admin_bp)
