import { createBrowserRouter } from "react-router-dom";
import App from "../App";
import HomePage from "../pages/HomePage";
import DocsPage from "../pages/DocsPage";


export const router = createBrowserRouter(
    [
        {
            path: "/",
            element: <App/>,
            children: [
                {path: "", element: <HomePage/>},
                {path: "docs", element: <DocsPage/>},
            ]
        }
    ]
)