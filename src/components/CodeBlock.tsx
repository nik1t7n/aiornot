import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { a11yLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';

interface Props {
    code: string;
}

const CodeBlock = ({ code }: Props) => {
    const codeClasses = "rounded-lg shadow-lg bg-gray-100 p-4";
  return (
    <div className={``}>
      <SyntaxHighlighter language="python" style={a11yLight} className={`${codeClasses}`}>
        {code}
      </SyntaxHighlighter>
    </div>
  );
};

export default CodeBlock;
