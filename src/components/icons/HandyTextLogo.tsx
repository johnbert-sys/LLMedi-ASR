import React from "react";
import logoUrl from "../../assets/llmedi-logo.png";

// llmedi wordmark, rendered from the original brand PNG (stethoscope glyph +
// "llmedi"). The file name is kept so upstream imports stay unchanged.
//
// The artwork is solid black on transparency, which would be invisible on a
// dark background, so it is used as a CSS mask and filled with the theme's
// --color-logo-stroke instead of being drawn as an image.
const ASPECT = 1810 / 430;

const HandyTextLogo = ({
  width,
  height,
  className,
}: {
  width?: number;
  height?: number;
  className?: string;
}) => {
  const resolvedWidth = width ?? (height ? height * ASPECT : 240);
  const resolvedHeight = height ?? resolvedWidth / ASPECT;

  return (
    <div
      role="img"
      aria-label="llmedi"
      className={className}
      style={{
        width: resolvedWidth,
        height: resolvedHeight,
        backgroundColor: "var(--color-logo-stroke)",
        maskImage: `url(${logoUrl})`,
        WebkitMaskImage: `url(${logoUrl})`,
        maskSize: "contain",
        WebkitMaskSize: "contain",
        maskRepeat: "no-repeat",
        WebkitMaskRepeat: "no-repeat",
        maskPosition: "center",
        WebkitMaskPosition: "center",
      }}
    />
  );
};

export default HandyTextLogo;
